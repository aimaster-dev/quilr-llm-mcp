"""Test doubles: an in-memory downstream that records every request it receives."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
from starlette.applications import Starlette

from mcp_gateway.config import GatewaySettings
from mcp_gateway.proxy import create_app

DOWNSTREAM_URL = "http://downstream.test/mcp"
SSE_HEADERS = {"content-type": "text/event-stream", "cache-control": "no-cache"}

Responder = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


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
        GatewaySettings(downstream_url=DOWNSTREAM_URL, **settings), transport=downstream.transport
    )


@asynccontextmanager
async def gateway_client(downstream: Downstream, **settings: Any) -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client wired to an in-memory gateway, which is wired to `downstream`."""
    app = make_app(downstream, **settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway.test") as client,
    ):
        yield client


def asgi_scope(method: str, headers: list[tuple[bytes, bytes]]) -> dict[str, Any]:
    """A raw ASGI HTTP scope for `/mcp`, for tests that drive the app message by message."""
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
        "headers": [(b"host", b"gateway.test"), *headers],
        "client": ("127.0.0.1", 50000),
        "server": ("gateway.test", 80),
    }
