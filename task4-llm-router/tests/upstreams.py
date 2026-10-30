"""Scriptable fake upstreams for router tests, built on httpx.MockTransport."""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
import httpx
from anyio.abc import SocketAttribute, SocketStream

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


# -- a real HTTP/1.1 server with scripted connection behaviour -----------------------------------

#: What the server does with one request: answer it, close the connection without sending a byte,
#: or send the status line and headers and then close mid-body.
Action = Literal["ok", "close", "headers-then-close"]


@dataclass
class ScriptedServer:
    """Records ``(connection number, request number on that connection)`` for every request."""

    url: str
    script: Callable[[int, int], Action]
    requests: list[tuple[int, int]] = field(default_factory=list)
    connections: int = 0


@asynccontextmanager
async def scripted_server(script: Callable[[int, int], Action]) -> AsyncIterator[ScriptedServer]:
    """Serve on 127.0.0.1 (ephemeral port), keeping connections alive until the script says not to.

    Lets tests reproduce, deterministically, a provider closing a keep-alive connection at the
    moment the gateway reuses it.
    """
    listener = await anyio.create_tcp_listener(local_host="127.0.0.1")
    port = listener.extra(SocketAttribute.local_port)  # noqa: S610 (anyio, not Django)
    server = ScriptedServer(f"http://127.0.0.1:{port}/v1", script)
    body = json.dumps(completion_body("from the scripted server")).encode()
    head = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n" % len(body)

    async def handle(stream: SocketStream) -> None:
        connection = server.connections
        server.connections += 1
        buffer = b""
        async with stream:
            for number in itertools.count():
                try:
                    while b"\r\n\r\n" not in buffer:
                        buffer += await stream.receive()
                    headers, _, buffer = buffer.partition(b"\r\n\r\n")
                    length = int(re.search(rb"(?i)content-length:\s*(\d+)", headers).group(1))
                    while len(buffer) < length:
                        buffer += await stream.receive()
                except (anyio.EndOfStream, anyio.BrokenResourceError, OSError):
                    return  # the client closed the connection
                buffer = buffer[length:]
                server.requests.append((connection, number))
                match server.script(connection, number):
                    case "close":
                        return
                    case "headers-then-close":
                        await stream.send(head + body[:10])
                        return
                    case "ok":
                        await stream.send(head + body)

    async with anyio.create_task_group() as tg:
        tg.start_soon(listener.serve, handle)
        try:
            yield server
        finally:
            tg.cancel_scope.cancel()
