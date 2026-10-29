"""Scriptable fake upstreams for router tests, built on httpx.MockTransport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import anyio
import httpx

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def completion_body(content: str = "hello", *, total_tokens: int | None = 42, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "model": "upstream-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        **extra,
    }
    if total_tokens is not None:
        body["usage"] = {
            "prompt_tokens": 2,
            "completion_tokens": total_tokens - 2,
            "total_tokens": total_tokens,
        }
    return body


class Recorder:
    """Wraps a handler, recording requests and whether each call finished or was cancelled."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.completed = 0
        self.cancelled = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        try:
            response = await self.handler(request)
        except anyio.get_cancelled_exc_class():
            self.cancelled += 1
            raise
        self.completed += 1
        return response

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def json(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def ok(content: str = "hello", *, total_tokens: int | None = 42, delay: float = 0.0) -> Handler:
    async def handler(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(delay)
        return httpx.Response(200, json=completion_body(content, total_tokens=total_tokens))

    return handler


def respond(
    status: int,
    *,
    json_body: Any = None,
    text: str | None = None,
    headers: dict[str, str] | None = None,
    delay: float = 0.0,
) -> Handler:
    async def handler(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(delay)
        if json_body is not None:
            return httpx.Response(status, json=json_body, headers=headers)
        return httpx.Response(status, text=text or "", headers=headers)

    return handler


def connection_refused() -> Handler:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connection refused", request=request)

    return handler


class DripStream(httpx.AsyncByteStream):
    """A response body that arrives one small chunk at a time."""

    def __init__(self, chunks: int, interval: float) -> None:
        self.chunks = chunks
        self.interval = interval
        self.sent = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(self.chunks):
            await anyio.sleep(self.interval)
            self.sent += 1
            yield b" "

    async def aclose(self) -> None:
        self.closed = True


def drip(stream: DripStream) -> Handler:
    """Headers immediately, then a body that trickles in (never trips a per-read timeout)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=stream)

    return handler
