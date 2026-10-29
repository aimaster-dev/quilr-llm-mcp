"""Drive an ASGI app directly and observe every message it sends.

``httpx.ASGITransport`` collects the whole response body before returning it, so it cannot show
*when* a chunk was written, or simulate a client that disconnects mid-stream. This driver can.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream
from starlette.types import ASGIApp, Message


class ASGIResponse:
    def __init__(self, messages: MemoryObjectReceiveStream[Message], disconnect: anyio.Event) -> None:
        self._messages = messages
        self._disconnect = disconnect
        self.status = 0
        self.headers: dict[str, str] = {}
        self.complete = False

    async def start(self) -> None:
        message = await self._messages.receive()
        assert message["type"] == "http.response.start"
        self.status = message["status"]
        self.headers = {k.decode(): v.decode() for k, v in message["headers"]}

    async def iter_body(self) -> AsyncIterator[bytes]:
        """Yield each ``http.response.body`` payload as soon as the app sends it."""
        async for message in self._messages:
            assert message["type"] == "http.response.body"
            if message.get("body"):
                yield message["body"]
            if not message.get("more_body", False):
                self.complete = True
                return

    async def read(self) -> bytes:
        return b"".join([chunk async for chunk in self.iter_body()])

    async def json(self) -> Any:
        return json.loads(await self.read())

    def disconnect(self) -> None:
        """Simulate the client closing the connection."""
        self._disconnect.set()


@asynccontextmanager
async def asgi_request(
    app: ASGIApp, path: str, body: Any, headers: dict[str, str] | None = None, method: str = "POST"
) -> AsyncIterator[ASGIResponse]:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    sent_body = False
    disconnect = anyio.Event()
    send_stream, receive_stream = anyio.create_memory_object_stream[Message](math.inf)

    async def receive() -> Message:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        await send_stream.send(message)

    request_headers = {"content-type": "application/json", "content-length": str(len(raw)), **(headers or {})}
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in request_headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
        "state": {},
    }

    async def run_app() -> None:
        async with send_stream:
            await app(scope, receive, send)

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_app)
        response = ASGIResponse(receive_stream, disconnect)
        with anyio.fail_after(10):
            await response.start()
        try:
            yield response
        finally:
            disconnect.set()  # let the app's disconnect listener finish if it is still running
