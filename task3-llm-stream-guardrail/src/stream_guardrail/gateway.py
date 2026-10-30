"""OpenAI-compatible gateway: proxies ``POST /v1/chat/completions`` and redacts PII in flight.

Streaming path (``"stream": true``)::

    upstream bytes -> SSEParser -> ChatStreamRedactor -> SSE bytes -> client

Every upstream read is parsed, redacted and written to the client before the next read. The gateway
holds one partial SSE line plus a few characters per choice, never the response.

The endpoint is a plain ASGI app so that it watches for the client going away during the whole
request: while the upstream is connecting and thinking, while a non-streaming answer is read, and
while a stream is relayed. A disconnect cancels that work and closes the upstream request with it.
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
from anyio.lowlevel import checkpoint
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from stream_guardrail.completions import ChatStreamRedactor, redact_completion
from stream_guardrail.config import Settings
from stream_guardrail.sse import SSEError, SSEEvent, SSEParser

logger = logging.getLogger("stream_guardrail.gateway")

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_UPSTREAM_ERROR_PEEK = 64 << 10
_SSE_HEADERS = [
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache"),
    (b"x-accel-buffering", b"no"),
]


def json_response(content: Any, status: int = 200, headers: dict[str, str] | None = None) -> Response:
    """A JSON response with ASCII escapes.

    A lone surrogate from upstream (``"\\ud83d"``) is valid JSON and decodes fine, but cannot be
    encoded as UTF-8; escaped, it goes through.
    """
    body = json.dumps(content, ensure_ascii=True, separators=(",", ":"))
    return Response(body, status_code=status, headers=headers, media_type="application/json")


@dataclass
class GatewayError(Exception):
    """A sanitized, OpenAI-style error. Upstream bodies and exception text never reach the client."""

    status: int
    code: str
    message: str
    type: str = "upstream_error"
    headers: dict[str, str] = field(default_factory=dict)
    upstream_code: str | None = None
    param: str | None = None

    def payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "message": self.message,
            "type": self.type,
            "param": self.param,
            "code": self.code,
        }
        if self.upstream_code:
            error["upstream_code"] = self.upstream_code
        return {"error": error}

    def response(self) -> Response:
        return json_response(self.payload(), self.status, self.headers)

    def event(self) -> SSEEvent:
        return SSEEvent(data=json.dumps(self.payload(), separators=(",", ":")))


def _timeout_error() -> GatewayError:
    return GatewayError(504, "upstream_timeout", "The upstream provider did not respond in time.")


def _unreachable_error() -> GatewayError:
    return GatewayError(502, "upstream_unavailable", "The upstream provider could not be reached.")


def _internal_error() -> GatewayError:
    return GatewayError(500, "internal_error", "The gateway failed to process the request.", "server_error")


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
    if status < 400:  # a redirect (never followed) or another status that is not a success
        return GatewayError(502, "upstream_error", "The upstream provider returned an unexpected status.")
    return GatewayError(502, "upstream_error", "The upstream provider returned an error.")


def _upstream_code(body: bytes) -> str | None:
    """Pass through the provider's machine-readable error code (e.g. ``model_not_found``) only."""
    try:
        error = json.loads(body).get("error")
        code = error.get("code") or error.get("type")
    except (ValueError, RecursionError, AttributeError):
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


async def _close(response: httpx.Response) -> None:
    """Close an upstream response, even from a cancelled task (the client went away)."""
    with anyio.CancelScope(shield=True):
        await response.aclose()


class Gateway:
    """The ``POST /v1/chat/completions`` endpoint, as an ASGI app."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        redactor_factory: Callable[[], ChatStreamRedactor] = ChatStreamRedactor,
    ) -> None:
        self.settings = settings
        self.client = client
        self.redactor_factory = redactor_factory

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        started = time.perf_counter()
        try:
            raw = await self._read_request(request)
        except ClientDisconnect:
            logger.info("client disconnected while sending the request")
            return
        except GatewayError as exc:
            logger.warning("request rejected: status=%d code=%s", exc.status, exc.code)
            await exc.response()(scope, receive, send)
            return
        # The body has been read, so receive() now only reports the disconnect. One task watches
        # for it for as long as the request runs; whichever side finishes first cancels the other.
        # Starlette's StreamingResponse watches only while streaming, and only on ASGI < 2.4.
        async with anyio.create_task_group() as tg:

            async def watch_disconnect() -> None:
                while (await receive())["type"] != "http.disconnect":
                    pass
                logger.info("client disconnected: request cancelled, upstream closed")
                tg.cancel_scope.cancel()

            tg.start_soon(watch_disconnect)
            await self._respond(request, raw, started, send)
            tg.cancel_scope.cancel()

    async def _respond(self, request: Request, raw: bytes, started: float, send: Send) -> None:
        try:
            stream = self._check_request(raw)
            upstream = await self._send(request, raw, stream)
            if stream and upstream.headers.get("content-type", "").startswith("text/event-stream"):
                await self._stream(upstream, started, send)
                return
            response = await self._complete(upstream, started)
        except GatewayError as exc:
            logger.warning("request failed: status=%d code=%s", exc.status, exc.code)
            response = exc.response()
        except Exception:
            logger.exception("request failed: unexpected error")
            response = _internal_error().response()
        with contextlib.suppress(OSError):  # the client went away
            await response(request.scope, request.receive, send)

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
    def _check_request(raw: bytes) -> bool:
        """Refuse what the gateway cannot redact; return whether the client asked for a stream."""
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):  # RecursionError: nested too deeply to parse
            raise GatewayError(
                400, "invalid_json", "Request body must be JSON.", "invalid_request_error"
            ) from None
        if not isinstance(body, dict):
            raise GatewayError(
                400, "invalid_request", "Request body must be a JSON object.", "invalid_request_error"
            )
        # Fail closed on output that bypasses text redaction: logprobs repeat every generated
        # token (with its bytes and alternatives), and audio speaks the answer.
        for name in ("logprobs", "top_logprobs"):
            if body.get(name):
                raise GatewayError(
                    400,
                    "logprobs_not_supported",
                    "logprobs and top_logprobs are not supported: token-level output cannot be redacted.",
                    "invalid_request_error",
                    param=name,
                )
        modalities = body.get("modalities")
        if body.get("audio") or (isinstance(modalities, list) and "audio" in modalities):
            raise GatewayError(
                400,
                "audio_not_supported",
                "Audio output is not supported: spoken output cannot be redacted.",
                "invalid_request_error",
                param="modalities",
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
        if 200 <= response.status_code < 300:
            return response
        body = b""
        try:
            body = await _read_limited(response.aiter_bytes(), _UPSTREAM_ERROR_PEEK)
        except (httpx.HTTPError, _BodyTooLarge):
            pass
        finally:
            await _close(response)
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
            await _close(upstream)
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            raise GatewayError(
                502, "upstream_invalid_response", "The upstream response is not valid JSON."
            ) from None
        try:
            body, redactions = redact_completion(body)
        except RecursionError:
            raise GatewayError(
                502, "upstream_invalid_response", "The upstream response is nested too deeply."
            ) from None
        logger.info(
            "completion relayed: stream=false redactions=%d duration_ms=%.1f",
            redactions,
            (time.perf_counter() - started) * 1000,
        )
        return json_response(body, upstream.status_code)

    async def _stream(self, upstream: httpx.Response, started: float, send: Send) -> None:
        """Send the redacted event stream. Failures after the status line are reported in-band."""
        events = self._relay(upstream, started)
        try:
            await send({"type": "http.response.start", "status": 200, "headers": _SSE_HEADERS})
            async for chunk in events:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
                # A server's send() usually returns without yielding. On asyncio's Proactor loop
                # (Windows) only the first write of a burst then reaches the socket, and the rest
                # waits until the whole upstream read has been redacted. Letting the loop run
                # after each event puts every event on the wire as soon as it is ready.
                await checkpoint()
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except OSError:
            logger.info("client disconnected during the stream")
        except Exception:
            logger.exception("stream failed: unexpected error while sending")
        finally:
            with anyio.CancelScope(shield=True):
                await events.aclose()
            await _close(upstream)  # also when the relay never started

    async def _relay(self, upstream: httpx.Response, started: float) -> AsyncGenerator[bytes, None]:
        parser = SSEParser(self.settings.max_event_size)
        redactor = self.redactor_factory()
        error: GatewayError | None = None
        complete: list[SSEEvent] = []  # events that were whole before the stream went bad
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
            except SSEError as exc:
                complete = exc.events  # e.g. good events in the same read as an oversized one
                error = GatewayError(
                    502, "upstream_invalid_stream", "The upstream sent a malformed event stream."
                )
            except Exception:
                # Last resort: never let a bug end the response with a broken chunked encoding.
                logger.exception("stream failed: unexpected error while relaying")
                error = GatewayError(
                    500, "internal_error", "The gateway failed while relaying the response.", "server_error"
                )
            if error is not None:
                # Deliver the events that were complete, release what was held (redacted as
                # end-of-text), then report the failure in-band: the 200 status line has already
                # been sent. If the redactor itself failed, its held text is dropped, not risked.
                logger.warning("stream aborted: code=%s", error.code)
                tail = b""
                with contextlib.suppress(Exception):
                    out = [e for event in complete for e in redactor.process(event)]
                    tail = b"".join(e.encode() for e in [*out, *redactor.finish()])
                yield tail + error.event().encode()
        finally:
            await _close(upstream)
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
        return json_response({"status": "ok"})

    async def internal_error(request: Request, exc: Exception) -> Response:
        # Anything that escaped the endpoint's own handling: the OpenAI error shape, no details.
        # Starlette logs the exception and re-raises it for the server.
        return _internal_error().response()

    return Starlette(
        routes=[
            Route("/v1/chat/completions", gateway, methods=["POST"]),
            Route("/healthz", healthz, methods=["GET"]),
        ],
        exception_handlers={Exception: internal_error},
        lifespan=lifespan,
    )
