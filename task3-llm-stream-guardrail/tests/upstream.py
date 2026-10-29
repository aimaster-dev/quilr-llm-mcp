"""Scripted upstream provider for gateway tests, plugged in through ``httpx.MockTransport``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from stream_guardrail.config import Settings

UPSTREAM_BASE = "http://upstream.test/v1"
SETTINGS = Settings(upstream_base_url=UPSTREAM_BASE, upstream_api_key="sk-gateway-secret")

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def completion_chunk(index: int = 0, finish: str | None = None, **fields: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "mock-1",
        "choices": [{"index": index, "delta": fields, "finish_reason": finish}],
    }


def sse(*payloads: Any) -> bytes:
    """Encode payloads as SSE ``data:`` events; strings are sent as-is (e.g. ``[DONE]``)."""
    return b"".join(
        b"data: " + (p if isinstance(p, str) else json.dumps(p)).encode() + b"\r\n\r\n" for p in payloads
    )


def content_stream(*pieces: str) -> bytes:
    payloads: list[Any] = [completion_chunk(role="assistant", content="")]
    payloads += [completion_chunk(content=piece) for piece in pieces]
    payloads += [completion_chunk(finish="stop"), "[DONE]"]
    return sse(*payloads)


def split_every(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


async def iterate(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def sse_response(chunks: list[bytes] | AsyncIterator[bytes], status: int = 200) -> httpx.Response:
    body = iterate(chunks) if isinstance(chunks, list) else chunks
    return httpx.Response(status, headers={"content-type": "text/event-stream"}, content=body)


class RecordingUpstream:
    """Records the requests it receives and answers with ``handler``."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self.handler(request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))
