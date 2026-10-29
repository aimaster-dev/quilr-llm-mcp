"""In-process helpers for tests, benchmarks and demos.

* ``serve(app)`` runs an ASGI app on a real uvicorn server on an ephemeral port.
* ``StreamingASGITransport`` is ``httpx.ASGITransport`` without its buffering. httpx's transport
  waits for the whole response body before returning, so it cannot stream. This one hands each
  body chunk to the client as soon as the app sends it, applies backpressure, and turns closing the
  response into an ``http.disconnect``.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import anyio
import httpx
import uvicorn
from anyio.streams.memory import MemoryObjectReceiveStream
from starlette.types import ASGIApp, Message


class _InProcessServer(uvicorn.Server):
    """A uvicorn server that signals when startup is over and leaves signal handlers alone."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.startup_done = anyio.Event()

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().startup(sockets=sockets)
        finally:
            self.startup_done.set()


@asynccontextmanager
async def serve(app: ASGIApp, host: str = "127.0.0.1") -> AsyncIterator[str]:
    """Run ``app`` on uvicorn on a free port for the duration of the block; yield its base URL."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level="warning", log_config=None, timeout_graceful_shutdown=5)
    server = _InProcessServer(config)
    async with anyio.create_task_group() as tg:
        tg.start_soon(server.serve, [sock])
        with anyio.fail_after(10):
            await server.startup_done.wait()
        if not server.started:
            raise RuntimeError("uvicorn failed to start")
        try:
            yield f"http://{host}:{port}"
        finally:
            server.should_exit = True


class _StreamingBody(httpx.AsyncByteStream):
    def __init__(
        self, messages: MemoryObjectReceiveStream[Message], task: asyncio.Task[None], disconnect: anyio.Event
    ) -> None:
        self._messages = messages
        self._task = task
        self._disconnect = disconnect

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for message in self._messages:
            if message["type"] != "http.response.body":
                continue
            if message.get("body"):
                yield message["body"]
            if not message.get("more_body", False):
                return

    async def aclose(self) -> None:
        self._disconnect.set()  # the app sees http.disconnect, like a closed socket
        self._messages.close()
        with anyio.CancelScope(shield=True):
            with anyio.move_on_after(5):
                await asyncio.wait([self._task])
            if not self._task.done():
                self._task.cancel()
            elif not self._task.cancelled():
                self._task.exception()  # an app that failed on a closed client is not news


class StreamingASGITransport(httpx.AsyncBaseTransport):
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        disconnect = anyio.Event()
        # A one-slot buffer gives backpressure: the app can be at most one message ahead.
        send_stream, receive_stream = anyio.create_memory_object_stream[Message](1)
        request_sent = False

        async def receive() -> Message:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            try:
                await send_stream.send(message)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                raise OSError("client disconnected") from None

        scope: dict[str, Any] = {
            "type": "http",
            # Same as uvicorn: servers below 2.4 deliver http.disconnect through receive(), which
            # lets Starlette's StreamingResponse cancel promptly.
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": request.method,
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "root_path": "",
            "headers": [(k.lower(), v) for k, v in request.headers.raw],
            "client": ("127.0.0.1", 50000),
            "server": (request.url.host, request.url.port or 80),
        }

        async def run() -> None:
            async with send_stream:
                await self.app(scope, receive, send)

        task = asyncio.get_running_loop().create_task(run())
        try:
            start = await receive_stream.receive()
        except anyio.EndOfStream:
            receive_stream.close()
            await task  # the app failed before responding: surface its exception
            raise RuntimeError("ASGI app returned without sending a response") from None
        headers = [(bytes(k), bytes(v)) for k, v in start.get("headers", [])]
        return httpx.Response(
            start["status"],
            headers=headers,
            stream=_StreamingBody(receive_stream, task, disconnect),
            request=request,
        )
