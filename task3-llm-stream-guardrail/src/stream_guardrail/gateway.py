"""OpenAI-compatible gateway: proxies ``POST /v1/chat/completions`` and redacts PII in flight.

Streaming path (``"stream": true``)::

    upstream bytes -> SSEParser -> ChatStreamRedactor -> SSE bytes -> client

Every upstream read is parsed, redacted and written to the client before the next read. The gateway
holds one partial SSE line plus a few characters per choice, never the response. When the client
disconnects, the relay is cancelled and the upstream request is closed with it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from stream_guardrail.completions import ChatStreamRedactor, redact_completion
from stream_guardrail.config import Settings
from stream_guardrail.sse import SSEError, SSEEvent, SSEParser

logger = logging.getLogger("stream_guardrail.gateway")

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_UPSTREAM_ERROR_PEEK = 64 << 10


@dataclass
class GatewayError(Exception):
    """A sanitized, OpenAI-style error. Upstream bodies and exception text never reach the client."""

    status: int
    code: str
    message: str
    type: str = "upstream_error"
    headers: dict[str, str] = field(default_factory=dict)
    upstream_code: str | None = None

    def payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {"message": self.message, "type": self.type, "code": self.code}
        if self.upstream_code:
            error["upstream_code"] = self.upstream_code
        return {"error": error}

    def response(self) -> JSONResponse:
        return JSONResponse(self.payload(), status_code=self.status, headers=self.headers)

    def event(self) -> SSEEvent:
        return SSEEvent(data=json.dumps(self.payload(), separators=(",", ":")))


def _timeout_error() -> GatewayError:
    return GatewayError(504, "upstream_timeout", "The upstream provider did not respond in time.")


def _unreachable_error() -> GatewayError:
    return GatewayError(502, "upstream_unavailable", "The upstream provider could not be reached.")


def _upstream_status_error(response: httpx.Response, body: bytes) -> GatewayError:
    status = response.status_code
    if status == 429:
        retry_after = response.headers.get("retry-after", "")
        headers = {"retry-after": retry_after} if _SAFE_TOKEN.fullmatch(retry_after) else {}
        return GatewayError(
            429,
            "rate_limited",
            "The upstream provider is rate limiting requests.",
            "rate_limit_error",
            headers,
        )
    if status in (401, 403):
        # The gateway's provider credential was rejected. That is not the client's fault.
        return GatewayError(502, "upstream_auth_failed", "The gateway could not authenticate upstream.")
    if status in (408, 504):
        return _timeout_error()
    if status in (400, 404, 409, 413, 415, 422):
        return GatewayError(
            status,
            "upstream_rejected_request",
            "The upstream provider rejected the request.",
            "invalid_request_error",
            upstream_code=_upstream_code(body),
        )
    return GatewayError(502, "upstream_error", "The upstream provider returned an error.")


def _upstream_code(body: bytes) -> str | None:
    """Pass through the provider's machine-readable error code (e.g. ``model_not_found``) only."""
    try:
        error = json.loads(body).get("error")
        code = error.get("code") or error.get("type")
    except (ValueError, AttributeError):
        return None
    return code if isinstance(code, str) and _SAFE_TOKEN.fullmatch(code) else None


class _BodyTooLarge(Exception):
    pass


async def _read_limited(chunks: AsyncIterator[bytes], limit: int) -> bytes:
    parts: list[bytes] = []
    size = 0
    async for chunk in chunks:
        size += len(chunk)
        if size > limit:
            raise _BodyTooLarge
        parts.append(chunk)
    return b"".join(parts)


class EventStreamResponse(StreamingResponse):
    """A StreamingResponse that always watches for client disconnects.

    Starlette only does this for servers that speak ASGI spec < 2.4. On newer servers a disconnect
    shows up only when the next ``send()`` fails, and that can be a long time away while the model
    is thinking. Here ``receive()`` is always watched, so the relay (and with it the upstream
    request) is cancelled as soon as the client goes away. The body generator is always closed.
    """

    def __init__(self, events: AsyncGenerator[bytes, None], headers: dict[str, str]) -> None:
        super().__init__(events, media_type="text/event-stream", headers=headers)
        self._events = events

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with anyio.create_task_group() as tg:

                async def stream() -> None:
                    with contextlib.suppress(OSError):  # the client went away mid-write
                        await self.stream_response(send)
                    tg.cancel_scope.cancel()

                tg.start_soon(stream)
                await self.listen_for_disconnect(receive)
                tg.cancel_scope.cancel()
        finally:
            with anyio.CancelScope(shield=True):
                await self._events.aclose()


class Gateway:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        redactor_factory: Callable[[], ChatStreamRedactor] = ChatStreamRedactor,
    ) -> None:
        self.settings = settings
        self.client = client
        self.redactor_factory = redactor_factory

    async def chat_completions(self, request: Request) -> Response:
        started = time.perf_counter()
        try:
            raw = await self._read_request(request)
            stream = self._wants_stream(raw)
            upstream = await self._send(request, raw, stream)
            content_type = upstream.headers.get("content-type", "")
            if stream and content_type.startswith("text/event-stream"):
                return EventStreamResponse(
                    self._relay(upstream, started),
                    headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                )
            return await self._complete(upstream, started)
        except GatewayError as exc:
            logger.warning("request failed: status=%d code=%s", exc.status, exc.code)
            return exc.response()

    async def _read_request(self, request: Request) -> bytes:
        limit = self.settings.max_request_bytes
        declared = request.headers.get("content-length", "")
        too_large = GatewayError(
            413, "request_too_large", "Request body is too large.", "invalid_request_error"
        )
        if declared.isdigit() and int(declared) > limit:
            raise too_large
        try:
            return await _read_limited(request.stream(), limit)
        except _BodyTooLarge:
            raise too_large from None

    @staticmethod
    def _wants_stream(raw: bytes) -> bool:
        try:
            body = json.loads(raw)
        except ValueError:
            raise GatewayError(
                400, "invalid_json", "Request body must be JSON.", "invalid_request_error"
            ) from None
        if not isinstance(body, dict):
            raise GatewayError(
                400, "invalid_request", "Request body must be a JSON object.", "invalid_request_error"
            )
        return body.get("stream") is True

    async def _send(self, request: Request, raw: bytes, stream: bool) -> httpx.Response:
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream" if stream else "application/json",
            # Compression would make the provider buffer before flushing tokens.
            "accept-encoding": "identity",
        }
        if self.settings.upstream_api_key:
            headers["authorization"] = f"Bearer {self.settings.upstream_api_key}"
        request_id = request.headers.get("x-request-id", "")
        if _SAFE_TOKEN.fullmatch(request_id):
            headers["x-request-id"] = request_id
        upstream_request = self.client.build_request(
            "POST", self.settings.chat_completions_url, content=raw, headers=headers
        )
        try:
            response = await self.client.send(upstream_request, stream=True)
        except httpx.TimeoutException:
            raise _timeout_error() from None
        except httpx.HTTPError:
            raise _unreachable_error() from None
        if response.status_code < 400:
            return response
        body = b""
        try:
            body = await _read_limited(response.aiter_bytes(), _UPSTREAM_ERROR_PEEK)
        except (httpx.HTTPError, _BodyTooLarge):
            pass
        finally:
            await response.aclose()
        logger.info("upstream returned status %d", response.status_code)
        raise _upstream_status_error(response, body)

    async def _complete(self, upstream: httpx.Response, started: float) -> Response:
        try:
            raw = await _read_limited(upstream.aiter_bytes(), self.settings.max_response_bytes)
        except httpx.TimeoutException:
            raise _timeout_error() from None
        except httpx.HTTPError:
            raise _unreachable_error() from None
        except _BodyTooLarge:
            raise GatewayError(
                502, "upstream_response_too_large", "The upstream response is too large."
            ) from None
        finally:
            await upstream.aclose()
        try:
            body = json.loads(raw)
        except ValueError:
            raise GatewayError(
                502, "upstream_invalid_response", "The upstream response is not valid JSON."
            ) from None
        body, redactions = redact_completion(body)
        logger.info(
            "completion relayed: stream=false redactions=%d duration_ms=%.1f",
            redactions,
            (time.perf_counter() - started) * 1000,
        )
        return JSONResponse(body, status_code=upstream.status_code)

    async def _relay(self, upstream: httpx.Response, started: float) -> AsyncGenerator[bytes, None]:
        parser = SSEParser(self.settings.max_event_size)
        redactor = self.redactor_factory()
        error: GatewayError | None = None
        first_event_ms: float | None = None
        try:
            try:
                async for raw in upstream.aiter_bytes():
                    # Each event is written as soon as it is redacted, even when one read held many.
                    for event in parser.feed(raw):
                        for out in redactor.process(event):
                            if first_event_ms is None:
                                first_event_ms = (time.perf_counter() - started) * 1000
                            yield out.encode()
                for event in parser.close():
                    for out in redactor.process(event):
                        yield out.encode()
                for out in redactor.finish():
                    yield out.encode()
            except httpx.TimeoutException:
                error = _timeout_error()
            except httpx.HTTPError:
                error = GatewayError(502, "upstream_stream_error", "The upstream stream was interrupted.")
            except SSEError:
                error = GatewayError(
                    502, "upstream_invalid_stream", "The upstream sent a malformed event stream."
                )
            if error is not None:
                # Release what was held (redacted as end-of-text), then report the failure in-band:
                # the 200 status line has already been sent.
                logger.warning("stream aborted: code=%s", error.code)
                yield b"".join(e.encode() for e in redactor.finish()) + error.event().encode()
        finally:
            with anyio.CancelScope(shield=True):
                await upstream.aclose()
            logger.info(
                "completion relayed: stream=true redactions=%d first_event_ms=%s duration_ms=%.1f",
                redactor.redactions,
                "n/a" if first_event_ms is None else f"{first_event_ms:.1f}",
                (time.perf_counter() - started) * 1000,
            )


def create_client(settings: Settings) -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        connect=settings.connect_timeout,
        read=settings.read_timeout,
        write=settings.write_timeout,
        pool=settings.pool_timeout,
    )
    return httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_keepalive_connections=20))


def create_app(
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    redactor_factory: Callable[[], ChatStreamRedactor] = ChatStreamRedactor,
) -> Starlette:
    """Build the ASGI app.

    ``client`` injects a transport (tests); the caller then owns it. ``redactor_factory`` builds
    the per-stream redactor (tests and benchmarks swap in instrumented or pass-through ones).
    """
    owns_client = client is None
    gateway = Gateway(settings, client or create_client(settings), redactor_factory)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owns_client:
                await gateway.client.aclose()

    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    return Starlette(
        routes=[
            Route("/v1/chat/completions", gateway.chat_completions, methods=["POST"]),
            Route("/healthz", healthz, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
