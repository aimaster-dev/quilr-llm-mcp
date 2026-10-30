"""Long-lived GET streams: their own connection pool and a per-subject cap, so they cannot starve requests."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import anyio
import httpx
import pytest
from starlette.applications import Starlette
from starlette.types import Receive, Scope, Send

from mcp_gateway.auth import Role
from mcp_gateway.config import GatewaySettings
from mcp_gateway.proxy import REQUEST_POOL_LIMITS, create_app

from .helpers import (
    SSE_HEADERS,
    TEST_SECRET,
    Downstream,
    ScriptedStream,
    asgi_scope,
    bearer,
    gateway_client,
    make_app,
    serve_in_thread,
)

pytestmark = pytest.mark.anyio

ADMIN_CALL = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "admin_reset_key", "arguments": {"key_id": "k-1"}},
}
STREAM_HEADERS = {"Accept": "text/event-stream"}


class HoldingDownstream:
    """A raw ASGI downstream: GET opens an SSE stream that stays open until the client leaves (like an
    MCP server's standalone stream); POST is answered with JSON at once."""

    def __init__(self) -> None:
        self.open_streams = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await receive()  # lifespan.startup
            await send({"type": "lifespan.startup.complete"})
            await receive()  # lifespan.shutdown
            await send({"type": "lifespan.shutdown.complete"})
            return
        if scope["method"] == "GET":
            self.open_streams += 1  # before the headers go out: a client that sees 200 sees this
            try:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")],
                    }
                )
                await send({"type": "http.response.body", "body": b": open\n\n", "more_body": True})
                while (await receive())["type"] != "http.disconnect":
                    pass
            finally:
                self.open_streams -= 1
            return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        request_id = json.loads(body)["id"]
        reply = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}}).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(reply)).encode())]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": reply})


# --- over real HTTP: the gateway's own connection pools -------------------------------------------


async def test_idle_get_streams_cannot_starve_requests_and_the_cap_answers_429() -> None:
    """The audit's reproduction: idle GET streams used to fill httpx's 100-connection pool, and
    an admin's tools/call then failed with 504. Here 11 viewers hold 10 streams each (the default
    cap), 110 in all, and requests still go through at once."""
    subjects = [f"viewer-{n}" for n in range(11)]
    downstream = HoldingDownstream()
    with serve_in_thread(downstream) as downstream_url:
        # A one-second pool wait, so a regression fails fast instead of hanging for 5 s per request.
        settings = GatewaySettings(
            jwt_secret=TEST_SECRET, downstream_url=f"{downstream_url}/mcp", connect_timeout=1
        )
        assert len(subjects) * settings.max_streams_per_subject > REQUEST_POOL_LIMITS.max_connections
        with serve_in_thread(create_app(settings)) as gateway_url:
            async with (
                httpx.AsyncClient(
                    base_url=gateway_url,
                    limits=httpx.Limits(max_connections=None),
                    timeout=10,
                    trust_env=False,
                ) as http,
                AsyncExitStack() as streams,
            ):
                for subject in subjects:
                    headers = {**bearer(Role.VIEWER, subject), **STREAM_HEADERS}
                    for _ in range(settings.max_streams_per_subject):
                        stream = await streams.enter_async_context(
                            http.stream("GET", "/mcp", headers=headers)
                        )
                        assert stream.status_code == 200
                    async with http.stream("GET", "/mcp", headers=headers) as refused:
                        assert refused.status_code == 429
                        assert json.loads(await refused.aread())["error"]["code"] == -32007
                assert downstream.open_streams == 110

                with anyio.fail_after(5):
                    admin = await http.post("/mcp", json=ADMIN_CALL, headers=bearer(Role.ADMIN, "root"))
                    viewer = await http.post(
                        "/mcp",
                        json={**ADMIN_CALL, "params": {"name": "get_weather"}},
                        headers=bearer(Role.VIEWER),
                    )

    assert admin.status_code == 200
    assert admin.json() == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    assert viewer.status_code == 200


# --- in memory: the per-subject cap ---------------------------------------------------------------


@asynccontextmanager
async def held_stream(app: Starlette, subject: str) -> AsyncIterator[dict[str, Any]]:
    """Open a GET stream on the in-memory gateway; the client leaves when the block exits."""
    started: dict[str, Any] = {}
    opened, leave = anyio.Event(), anyio.Event()
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await leave.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            started.update(message, body=b"")
            opened.set()
        else:
            started["body"] += message.get("body", b"")

    async with anyio.create_task_group() as tg:
        tg.start_soon(
            app, asgi_scope("GET", [(b"accept", b"text/event-stream")], subject=subject), receive, send
        )
        with anyio.fail_after(5):
            await opened.wait()
        try:
            yield started
        finally:
            leave.set()


def idle_sse(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        return httpx.Response(200, headers=SSE_HEADERS, stream=ScriptedStream(b": open\n\n", anyio.Event()))
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})


async def test_a_subject_over_its_stream_cap_gets_429_and_others_do_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="mcp_gateway.audit")
    downstream = Downstream(idle_sse)
    app = make_app(downstream, max_streams_per_subject=2)
    async with (
        app.router.lifespan_context(app),
        held_stream(app, "alice") as first,
        held_stream(app, "alice") as second,
        held_stream(app, "bob") as other_subject,
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway.test") as client,
    ):
        async with held_stream(app, "alice") as refused:
            pass
        allowed_post = await client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=bearer(Role.VIEWER, "alice")
        )

    assert [first["status"], second["status"], other_subject["status"]] == [200, 200, 200]
    assert refused["status"] == 429
    assert (b"content-type", b"application/json") in refused["headers"]
    assert json.loads(refused["body"]) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32007, "message": "Too many open streams: close one before opening another"},
    }
    assert allowed_post.status_code == 200  # the cap is on streams, not on requests
    assert sum(request.method == "GET" for request in downstream.requests) == 3  # the 429 never got there
    rejects = [json.loads(r.getMessage()) for r in caplog.records if '"decision":"reject"' in r.getMessage()]
    assert [(line["subject"], line["status"], line["code"]) for line in rejects] == [("alice", 429, -32007)]
    assert rejects[0]["reason"].startswith("stream_limit_exceeded")


async def test_a_closed_stream_frees_its_slot() -> None:
    downstream = Downstream(idle_sse)
    app = make_app(downstream, max_streams_per_subject=1)
    gateway = app.state.gateway
    async with app.router.lifespan_context(app):
        async with held_stream(app, "alice") as first:
            assert first["status"] == 200
            assert gateway.streams.open_streams("alice") == 1
        # The client left: the relay was torn down and the slot released.
        assert gateway.streams.open_streams("alice") == 0
        async with held_stream(app, "alice") as again:
            assert again["status"] == 200


def stream_that_ends(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers=SSE_HEADERS, stream=ScriptedStream(b": bye\n\n"))


def stream_refused(request: httpx.Request) -> httpx.Response:
    return httpx.Response(404)


def downstream_unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused")


@pytest.mark.parametrize(
    ("responder", "status"),
    [(stream_that_ends, 200), (stream_refused, 404), (downstream_unreachable, 502)],
)
async def test_streams_that_end_or_fail_free_their_slot(responder: Any, status: int) -> None:
    downstream = Downstream(responder)
    async with gateway_client(downstream, max_streams_per_subject=1) as gateway:
        statuses = [(await gateway.get("/mcp", headers=STREAM_HEADERS)).status_code for _ in range(3)]
    assert statuses == [status] * 3
    assert len(downstream.requests) == 3
