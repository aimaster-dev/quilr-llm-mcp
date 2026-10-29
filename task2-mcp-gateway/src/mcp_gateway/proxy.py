"""ASGI reverse proxy for the MCP Streamable HTTP transport.

The gateway exposes one endpoint, `/mcp`, and relays it to one downstream URL:

- POST carries a JSON-RPC message. The downstream answers with `application/json` or with a
  `text/event-stream`, and either is relayed to the client chunk by chunk, never buffered.
- GET opens the server-to-client SSE stream, DELETE terminates a session.

Every request is authenticated first. A POST body is then parsed, checked against the policy, and
either answered by the gateway (error or denial) or forwarded in canonical form.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anyio
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from mcp_gateway.audit import AuditRecord
from mcp_gateway.auth import AuthenticationError, Identity, Role, TokenVerifier, bearer_token
from mcp_gateway.config import GatewaySettings
from mcp_gateway.jsonrpc import (
    AUTHENTICATION_FAILED,
    DOWNSTREAM_TIMEOUT,
    DOWNSTREAM_UNAVAILABLE,
    INVALID_REQUEST,
    UNAUTHORIZED_TOOL_CALL,
    JsonRpcError,
    RequestId,
    parse_message,
)
from mcp_gateway.policy import authorize, check_routing_headers

logger = logging.getLogger("mcp_gateway.proxy")

MCP_PATH = "/mcp"
ALLOWED_METHODS = ("GET", "POST", "DELETE")
_BODYLESS_REASONS = {"GET": "sse_stream", "DELETE": "session_termination"}

RawHeaders = list[tuple[bytes, bytes]]

# RFC 9110 section 7.6.1: connection-specific headers that are meaningful for a single hop only.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
# Identity the gateway asserts to the downstream. Client-supplied headers with this prefix are
# dropped, so only the gateway can set them.
IDENTITY_HEADER_PREFIX = "x-gateway-"
SUBJECT_HEADER = "x-gateway-subject"
ROLE_HEADER = "x-gateway-role"
REQUEST_ID_HEADER = "x-gateway-request-id"

# - host and content-length are recomputed for the downstream request.
# - accept-encoding is replaced by "identity": a compressor would buffer SSE events.
# - The client's credentials stay at the gateway. Forwarding its token would make the downstream
#   accept a token that was not issued for it (the MCP spec forbids token passthrough).
DROPPED_REQUEST_HEADERS = HOP_BY_HOP_HEADERS | {
    "host",
    "content-length",
    "expect",
    "accept-encoding",
    "authorization",
    "cookie",
}
# The ASGI server sets its own date/server headers; the downstream's would duplicate them.
DROPPED_RESPONSE_HEADERS = HOP_BY_HOP_HEADERS | {"date", "server"}

_CONTENT_LENGTH = re.compile(r"[0-9]{1,19}")


def filter_headers(
    raw: Iterable[tuple[bytes, bytes]], drop: frozenset[str], drop_prefix: str | None = None
) -> RawHeaders:
    """Copy headers (lower-cased names) minus `drop`, `drop_prefix*` and those named by `Connection`."""
    headers = [(name.lower(), value) for name, value in raw]
    nominated = {
        token.strip().lower()
        for name, value in headers
        if name == b"connection"
        for token in value.decode("latin-1").split(",")
    }
    return [
        (name, value)
        for name, value in headers
        if (key := name.decode("latin-1")) not in drop
        and key not in nominated
        and not (drop_prefix and key.startswith(drop_prefix))
    ]


def error_response(error: JsonRpcError) -> Response:
    body = json.dumps(error.payload(), separators=(",", ":")).encode()
    return Response(body, status_code=error.http_status, media_type="application/json", headers=error.headers)


def check_content_type(value: str | None) -> None:
    """POST bodies must be `application/json`, optionally with `charset=utf-8`."""
    media_type, _, params = (value or "").partition(";")
    acceptable = media_type.strip().lower() == "application/json"
    for param in params.split(";"):
        key, _, charset = param.partition("=")
        if key.strip().lower() == "charset" and charset.strip().strip('"').lower() not in ("utf-8", "utf8"):
            acceptable = False
    if not acceptable:
        raise JsonRpcError(INVALID_REQUEST, "Content-Type must be application/json (UTF-8)", http_status=415)


async def read_body(request: Request, limit: int) -> bytes:
    """Read the request body, refusing anything larger than `limit` bytes without buffering it."""
    too_large = JsonRpcError(INVALID_REQUEST, f"Request body exceeds {limit} bytes", http_status=413)
    declared = request.headers.get("content-length")
    if declared is not None:
        if not _CONTENT_LENGTH.fullmatch(declared):
            raise JsonRpcError(INVALID_REQUEST, "Invalid Content-Length header")
        if int(declared) > limit:
            raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise too_large
    return bytes(body)


class StreamedResponse(Response):
    """Relay a downstream response as it arrives, and always close it afterwards.

    Starlette's `StreamingResponse` only watches for client disconnects on ASGI servers older than
    spec 2.4. This relay always does, so an idle SSE stream is torn down, and its downstream
    connection released, as soon as the client goes away.
    """

    def __init__(self, upstream: httpx.Response) -> None:
        # Response.__init__ is not called: it would add a Content-Length for an empty body.
        self.status_code = upstream.status_code
        self.background = None
        self.raw_headers = filter_headers(upstream.headers.raw, DROPPED_RESPONSE_HEADERS)
        self._upstream = upstream

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._cancel_on_disconnect, receive, tg.cancel_scope)
                await self._relay(send)
                tg.cancel_scope.cancel()
        finally:
            with anyio.CancelScope(shield=True):
                await self._upstream.aclose()

    async def _relay(self, send: Send) -> None:
        try:
            await send(
                {"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers}
            )
            async for chunk in self._upstream.aiter_raw():
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        except httpx.RequestError as exc:
            # Status and headers are already on the wire. The honest signal left is to end the
            # connection without completing the body, so the client sees a truncated response.
            logger.warning("downstream stream failed mid-response: %r", exc)
            return
        except OSError:
            return  # the client went away while we were sending
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    @staticmethod
    async def _cancel_on_disconnect(receive: Receive, scope: anyio.CancelScope) -> None:
        while (await receive())["type"] != "http.disconnect":
            pass
        scope.cancel()


@dataclass(frozen=True)
class Admitted:
    """A request the gateway decided to forward."""

    content: bytes | None
    rpc_id: RequestId | None
    reason: str


class Gateway:
    """The `/mcp` endpoint (an ASGI app): authenticates, validates, authorizes, forwards."""

    def __init__(self, settings: GatewaySettings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self._client = client
        self._verifier = TokenVerifier(
            secret=settings.jwt_secret,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            leeway=settings.jwt_leeway,
        )
        connect, read = settings.connect_timeout, settings.read_timeout
        self._timeout = httpx.Timeout(connect=connect, read=read, write=read, pool=connect)
        # GET is a long-lived server-to-client stream: silence on it is normal, not a failure.
        self._stream_timeout = httpx.Timeout(connect=connect, read=None, write=read, pool=connect)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # A plain ASGI app rather than a Starlette endpoint, so every HTTP method reaches `handle`
        # and gets a JSON-RPC shaped 405 instead of Starlette's plain-text one.
        response = await self.handle(Request(scope, receive))
        await response(scope, receive, send)

    async def handle(self, request: Request) -> Response:
        """Admit or refuse the request (one audit line either way), then forward it."""
        record = AuditRecord(request_id=uuid.uuid4().hex, http_method=request.method)
        try:
            identity = self.authenticate(request)
            record.subject, record.role = identity.subject, identity.role.value
            admitted = await self._admit(request, identity.role, record)
        except JsonRpcError as error:
            verdict = "deny" if error.code == UNAUTHORIZED_TOOL_CALL else "reject"
            record.emit(verdict, error.detail or error.message, status=error.http_status, code=error.code)
            return error_response(error)

        record.emit("allow", admitted.reason)
        try:
            return await self.forward(request, identity, record.request_id, admitted)
        except JsonRpcError as error:
            return error_response(error)

    def authenticate(self, request: Request) -> Identity:
        """Verify the bearer token, before anything else about the request is looked at."""
        try:
            return self._verifier.verify(bearer_token(request.headers.getlist("authorization")))
        except AuthenticationError as exc:
            raise JsonRpcError(
                AUTHENTICATION_FAILED,
                exc.description,
                http_status=exc.http_status,
                headers={"WWW-Authenticate": exc.www_authenticate},
                detail=f"authentication failed: {exc.reason}",
            ) from exc

    async def _admit(self, request: Request, role: Role, record: AuditRecord) -> Admitted:
        """Validate the request and apply the policy. Raises `JsonRpcError` to refuse it."""
        if request.method in _BODYLESS_REASONS:  # no JSON-RPC message, nothing to authorize
            return Admitted(content=None, rpc_id=None, reason=_BODYLESS_REASONS[request.method])
        if request.method != "POST":
            raise JsonRpcError(
                INVALID_REQUEST,
                "HTTP method not allowed",
                http_status=405,
                headers={"Allow": ", ".join(ALLOWED_METHODS)},
            )
        check_content_type(request.headers.get("content-type"))
        message = parse_message(await read_body(request, self.settings.max_body_bytes))
        record.method, record.rpc_id = message.method, message.id
        headers = [(name.decode("latin-1"), value.decode("latin-1")) for name, value in request.headers.raw]
        check_routing_headers(message, headers)

        decision = authorize(message, role)
        record.tool = decision.tool
        rpc_id = message.id if message.method is not None else None  # a client response is not a request
        if not decision.allowed:
            # HTTP 200: the exchange itself worked, and MCP clients surface a JSON-RPC error for
            # this one call (the official SDK raises MCPError(-32001)). A 401/403 would instead
            # look like a broken session or trigger a token refresh.
            raise JsonRpcError(
                UNAUTHORIZED_TOOL_CALL,
                "Unauthorized Tool Call",
                http_status=200,
                rpc_id=rpc_id,
                data={"requiredRole": Role.ADMIN.value},
                detail=decision.reason,
            )
        # Forward the canonical bytes, not the received ones: the downstream must see exactly the
        # message that was authorized.
        return Admitted(content=message.canonical, rpc_id=rpc_id, reason=decision.reason)

    async def forward(
        self, request: Request, identity: Identity, request_id: str, admitted: Admitted
    ) -> Response:
        """Send the request downstream and relay the response; transport failures become 502/504."""
        rpc_id = admitted.rpc_id
        headers = filter_headers(request.headers.raw, DROPPED_REQUEST_HEADERS, IDENTITY_HEADER_PREFIX)
        headers += [
            (b"accept-encoding", b"identity"),
            (SUBJECT_HEADER.encode(), identity.subject.encode()),
            (ROLE_HEADER.encode(), identity.role.value.encode()),
            (REQUEST_ID_HEADER.encode(), request_id.encode()),
        ]
        upstream_request = self._client.build_request(
            request.method,
            self.settings.downstream_url,
            headers=headers,
            content=admitted.content,  # httpx sets Content-Length from these bytes
            timeout=self._stream_timeout if request.method == "GET" else self._timeout,
        )
        try:
            upstream = await self._client.send(upstream_request, stream=True)
        except httpx.TimeoutException as exc:
            logger.warning("downstream %s timed out: %r", request.method, exc)
            raise JsonRpcError(
                DOWNSTREAM_TIMEOUT, "Downstream MCP server timed out", http_status=504, rpc_id=rpc_id
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("downstream %s failed: %r", request.method, exc)
            raise JsonRpcError(
                DOWNSTREAM_UNAVAILABLE, "Downstream MCP server is unavailable", http_status=502, rpc_id=rpc_id
            ) from exc
        logger.debug("%s %s -> %d", request.method, MCP_PATH, upstream.status_code)
        return StreamedResponse(upstream)


def create_app(settings: GatewaySettings, *, transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    """Build the gateway ASGI app. `transport` replaces the network (tests pass a mock downstream)."""
    # trust_env=False: HTTP(S)_PROXY and netrc from the environment must not reroute or add
    # credentials to downstream traffic.
    client = httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False)
    gateway = Gateway(settings, client)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    app = Starlette(routes=[Route(MCP_PATH, gateway)], lifespan=lifespan)
    app.state.gateway = gateway
    return app
