"""Test doubles (an in-memory downstream that records every request it receives), tokens, servers."""

from __future__ import annotations

import inspect
import json
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any
from wsgiref.simple_server import WSGIRequestHandler, make_server

import anyio
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.types import ASGIApp

if TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication

from mcp_gateway.auth import Role, mint_token
from mcp_gateway.config import DEFAULT_JWT_AUDIENCE, DEFAULT_JWT_ISSUER, GatewaySettings
from mcp_gateway.proxy import create_app

DOWNSTREAM_URL = "http://downstream.test/mcp"
SSE_HEADERS = {"content-type": "text/event-stream", "cache-control": "no-cache"}
TEST_SECRET = "test-only-secret-" + "0123456789abcdef" * 4

Responder = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


def token_for(role: Role | str, subject: str = "alice", **overrides: Any) -> str:
    """A valid token for the test gateway; `overrides` replace mint_token arguments."""
    arguments: dict[str, Any] = {
        "secret": TEST_SECRET,
        "subject": subject,
        "role": Role(role),
        "issuer": DEFAULT_JWT_ISSUER,
        "audience": DEFAULT_JWT_AUDIENCE,
        **overrides,
    }
    return mint_token(**arguments)


def bearer(role: Role | str, subject: str = "alice") -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for(role, subject)}"}


def jsonrpc_echo(request: httpx.Request) -> httpx.Response:
    """Answer like an MCP server in JSON-response mode: requests get `{"method": ...}` back."""
    if request.method != "POST":
        return httpx.Response(405)
    message = json.loads(request.content)
    if "method" not in message or "id" not in message:
        return httpx.Response(202)  # notifications and client responses get no body
    return httpx.Response(
        200, json={"jsonrpc": "2.0", "id": message["id"], "result": {"method": message["method"]}}
    )


class Downstream:
    """In-memory downstream MCP server (an httpx MockTransport) that records every request."""

    def __init__(self, responder: Responder = jsonrpc_echo) -> None:
        self.requests: list[httpx.Request] = []
        self.responder = responder
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        response = self.responder(request)
        if inspect.isawaitable(response):
            response = await response
        if isinstance(response.stream, ScriptedStream):
            return response
        # httpx pre-reads a Response built from bytes/json; a network response is a stream.
        return httpx.Response(
            response.status_code, headers=response.headers, stream=ScriptedStream(response.content)
        )

    @property
    def last(self) -> httpx.Request:
        assert self.requests, "the downstream was never called"
        return self.requests[-1]


class ScriptedStream(httpx.AsyncByteStream):
    """A downstream response body: yields bytes, waits on events, raises exceptions, records aclose()."""

    def __init__(self, *parts: bytes | anyio.Event | Exception) -> None:
        self.parts = parts
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            if isinstance(part, bytes):
                yield part
            elif isinstance(part, anyio.Event):
                await part.wait()
            else:
                raise part

    async def aclose(self) -> None:
        self.closed = True


def make_app(downstream: Downstream, **settings: Any) -> Starlette:
    return create_app(
        GatewaySettings(jwt_secret=TEST_SECRET, downstream_url=DOWNSTREAM_URL, **settings),
        transport=downstream.transport,
    )


@asynccontextmanager
async def gateway_client(
    downstream: Downstream, *, role: Role | str | None = Role.VIEWER, **settings: Any
) -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client wired to an in-memory gateway, which is wired to `downstream`.

    The client authenticates as `role` (subject "alice") unless `role` is None.
    """
    app = make_app(downstream, **settings)
    headers = bearer(role) if role is not None else {}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway.test", headers=headers
        ) as client,
    ):
        yield client


def asgi_scope(method: str, headers: list[tuple[bytes, bytes]], role: Role = Role.VIEWER) -> dict[str, Any]:
    """A raw ASGI HTTP scope for `/mcp`, for tests that drive the app message by message."""
    authorization = (b"authorization", f"Bearer {token_for(role)}".encode())
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"gateway.test"), authorization, *headers],
        "client": ("127.0.0.1", 50000),
        "server": ("gateway.test", 80),
    }


class _QuietWSGIRequestHandler(WSGIRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass


@contextmanager
def serve_wsgi_in_thread(app: WSGIApplication) -> Iterator[str]:
    """Serve a WSGI app with the stdlib `wsgiref` server on an ephemeral localhost port; yield its URL.

    `wsgiref`, like every CGI-style server, maps header names to `HTTP_*` keys with `-` turned into
    `_`, so `X-Gateway-Role` and `X_Gateway_Role` reach the app as the same key.
    """
    server = make_server("127.0.0.1", 0, app, handler_class=_QuietWSGIRequestHandler)
    thread = threading.Thread(target=server.serve_forever, name="wsgiref-test-server", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@contextmanager
def serve_in_thread(app: ASGIApp) -> Iterator[str]:
    """Serve `app` with uvicorn on an ephemeral localhost port in a background thread; yield its URL."""
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_config=None, log_level="warning", timeout_graceful_shutdown=2
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn-test-server", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
