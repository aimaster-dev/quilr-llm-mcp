"""Mock OpenAI-compatible LLM provider with configurable chunking and timing.

It streams a fixed text (by default one sprinkled with PII) as ``chat.completion.chunk`` events:

* ``first_token_delay``: simulated time to first token, before the role chunk;
* ``chunk_size`` characters per content delta, ``delay`` seconds apart;
* ``repeat``: repeat the text, generated lazily, for long-stream tests.

Per-request overrides go in a non-standard ``"mock"`` object in the request body, which the gateway
forwards untouched::

    {"model": "m", "stream": true, "messages": [...],
     "mock": {"chunk_size": 3, "delay_ms": 5, "first_token_delay_ms": 0, "repeat": 10,
              "text": "...", "echo": true}}

``echo`` streams back the last user message, so you can try redaction on your own input.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from stream_guardrail.logging_config import configure_logging

DEFAULT_TEXT = (
    "Sure! Here is the customer record you asked for. Name: Jane Roe. Email: jane.roe@example.com. "
    "Card on file: 4111 1111 1111 1111, backup card 5500-0000-0000-0004. SSN: 078-05-1120. "
    "Order number 1234567890123 and version 2.4.1 are not sensitive. Anything else?"
)

ChunkHook = Callable[[int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class MockSettings:
    text: str = DEFAULT_TEXT
    chunk_size: int = 4
    delay: float = 0.02
    first_token_delay: float = 0.1
    repeat: int = 1
    model: str = "mock-llm-1"
    api_key: str | None = field(default=None, repr=False)  # require this bearer token if set


@dataclass(slots=True)
class MockStats:
    requests: int = 0
    completed: int = 0  # streams that sent [DONE]
    cancelled: int = 0  # streams cut short by the client (e.g. the gateway) going away


_LIMITS = {
    "chunk_size": (int, 1, 1 << 20),
    "delay_ms": (float, 0, 10_000),
    "first_token_delay_ms": (float, 0, 60_000),
    "repeat": (int, 1, 1 << 20),
}


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}}, status
    )


class MockProvider:
    """``app`` is the ASGI app. ``before_chunk(k)`` is awaited before content delta ``k`` is sent;
    tests use it to hold the stream until the client has observed something."""

    def __init__(
        self,
        settings: MockSettings | None = None,
        before_chunk: ChunkHook | None = None,
        on_stream_end: Callable[[bool], None] | None = None,
    ) -> None:
        self.settings = settings or MockSettings()
        self.before_chunk = before_chunk
        self.on_stream_end = on_stream_end
        self.stats = MockStats()
        self.app = Starlette(
            routes=[
                Route("/v1/chat/completions", self.chat_completions, methods=["POST"]),
                Route("/healthz", self.healthz, methods=["GET"]),
            ]
        )

    async def healthz(self, request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def chat_completions(self, request: Request) -> Response:
        self.stats.requests += 1
        key = self.settings.api_key
        if key and request.headers.get("authorization") != f"Bearer {key}":
            return _error(401, "Invalid API key.", "invalid_api_key")
        try:
            body = json.loads(await request.body())
        except ValueError:
            return _error(400, "Body must be JSON.", "invalid_json")
        if not isinstance(body, dict):
            return _error(400, "Body must be a JSON object.", "invalid_request")
        try:
            settings = self._effective_settings(body)
        except ValueError as exc:
            return _error(400, str(exc), "invalid_mock_options")
        n = body.get("n", 1)
        if not isinstance(n, int) or not 1 <= n <= 8:
            return _error(400, "n must be an integer between 1 and 8.", "invalid_n")
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        if body.get("stream") is True:
            return StreamingResponse(
                self._stream(settings, n, include_usage),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache"},
            )
        await anyio.sleep(settings.first_token_delay)
        return JSONResponse(self._completion(settings, n))

    def _effective_settings(self, body: dict[str, Any]) -> MockSettings:
        options = body.get("mock") or {}
        if not isinstance(options, dict):
            raise ValueError("mock must be an object")
        changes: dict[str, Any] = {}
        for name, (kind, low, high) in _LIMITS.items():
            if name not in options:
                continue
            value = options[name]
            if isinstance(value, bool) or not isinstance(value, int | float) or not low <= value <= high:
                raise ValueError(f"mock.{name} must be a number between {low} and {high}")
            value = kind(value)
            if name.endswith("_ms"):
                changes[name.removesuffix("_ms")] = value / 1000
            else:
                changes[name] = value
        if "text" in options:
            if not isinstance(options["text"], str):
                raise ValueError("mock.text must be a string")
            changes["text"] = options["text"]
        elif options.get("echo"):
            changes["text"] = _last_user_message(body)
        return replace(self.settings, **changes)

    async def _stream(self, settings: MockSettings, n: int, include_usage: bool) -> AsyncIterator[bytes]:
        header = {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": settings.model,
        }

        def event(choices: list[dict[str, Any]], **extra: Any) -> bytes:
            payload = {**header, "choices": choices, **extra}
            return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"

        completed = False
        try:
            await anyio.sleep(settings.first_token_delay)
            role = {"role": "assistant", "content": ""}
            yield event([{"index": i, "delta": role, "finish_reason": None} for i in range(n)])
            k = 0
            text, size = settings.text, settings.chunk_size
            for _ in range(settings.repeat):
                for start in range(0, len(text), size):
                    if self.before_chunk is not None:
                        await self.before_chunk(k)
                    if k and settings.delay:
                        await anyio.sleep(settings.delay)
                    piece = text[start : start + size]
                    for i in range(n):
                        yield event([{"index": i, "delta": {"content": piece}, "finish_reason": None}])
                    k += 1
            yield event([{"index": i, "delta": {}, "finish_reason": "stop"} for i in range(n)])
            if include_usage:
                tokens = (len(text) * settings.repeat + 3) // 4 * n
                usage = {"prompt_tokens": 10, "completion_tokens": tokens, "total_tokens": tokens + 10}
                yield event([], usage=usage)
            yield b"data: [DONE]\n\n"
            completed = True
        finally:
            if completed:
                self.stats.completed += 1
            else:
                self.stats.cancelled += 1
            if self.on_stream_end is not None:
                self.on_stream_end(completed)

    def _completion(self, settings: MockSettings, n: int) -> dict[str, Any]:
        content = settings.text * settings.repeat
        return {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": settings.model,
            "choices": [
                {"index": i, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                for i in range(n)
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(content) // 4,
                "total_tokens": 10 + len(content) // 4,
            },
        }


def _last_user_message(body: dict[str, Any]) -> str:
    for message in reversed(body.get("messages") or []):
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ):
            return message["content"]
    return ""


def main(argv: Sequence[str] | None = None) -> None:
    """`stream-guardrail-mock` console script."""
    parser = argparse.ArgumentParser(
        prog="stream-guardrail-mock",
        description="Mock OpenAI-compatible provider that streams a canned, PII-laden answer. "
        "Set $MOCK_LLM_API_KEY to require a bearer token.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--chunk-size", type=int, default=4, help="characters per delta (default: 4)")
    parser.add_argument("--delay-ms", type=float, default=20, help="pause between deltas (default: 20)")
    parser.add_argument("--ttft-ms", type=float, default=100, help="time to first token (default: 100)")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to stream (default: a PII sample)")
    parser.add_argument("--log-level", help="DEBUG, INFO, WARNING or ERROR (default: INFO)")
    args = parser.parse_args(argv)

    level = configure_logging(args.log_level)
    settings = MockSettings(
        text=args.text,
        chunk_size=args.chunk_size,
        delay=args.delay_ms / 1000,
        first_token_delay=args.ttft_ms / 1000,
        api_key=os.environ.get("MOCK_LLM_API_KEY") or None,
    )
    provider = MockProvider(settings)
    uvicorn.run(provider.app, host=args.host, port=args.port, log_level=level.lower(), log_config=None)


if __name__ == "__main__":
    main()
