"""HTTP forwarding: what reaches the downstream, what comes back, and how failures surface."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import httpx
import pytest

from .helpers import (
    DOWNSTREAM_URL,
    SSE_HEADERS,
    Downstream,
    ScriptedStream,
    asgi_scope,
    gateway_client,
    make_app,
)

pytestmark = pytest.mark.anyio

PING = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
EVENT_1 = b'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
EVENT_2 = b'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{}}\n\n'
GATEWAY_ERROR_CODES = {502: -32004, 504: -32005}


async def post(client: httpx.AsyncClient, payload: Any, **headers: str) -> httpx.Response:
    return await client.post("/mcp", json=payload, headers=headers)


def assert_jsonrpc_error(response: httpx.Response, status: int, code: int) -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["error"]["code"] == code
    return body


async def test_post_is_forwarded_to_the_downstream(
    gateway: httpx.AsyncClient, downstream: Downstream
) -> None:
    response = await post(gateway, PING)

    assert response.status_code == 200
    assert response.json() == {"jsonrpc": "2.0", "id": 1, "result": {"method": "ping"}}
    forwarded = downstream.last
    assert forwarded.method == "POST"
    assert str(forwarded.url) == DOWNSTREAM_URL
    assert forwarded.headers["content-type"] == "application/json"
    assert json.loads(forwarded.content) == PING
    assert forwarded.headers["content-length"] == str(len(forwarded.content))


async def test_query_string_is_not_forwarded(gateway: httpx.AsyncClient, downstream: Downstream) -> None:
    await gateway.post("/mcp?target=http://evil.test", json=PING)
    assert str(downstream.last.url) == DOWNSTREAM_URL


async def test_response_status_headers_and_body_are_relayed(downstream: Downstream) -> None:
    downstream.responder = lambda _: httpx.Response(
        404,
        headers={"x-downstream": "yes", "server": "secret-stack/1.0"},
        json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Method not found"}},
    )
    async with gateway_client(downstream) as gateway:
        response = await post(gateway, PING)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == -32601
    assert response.headers["x-downstream"] == "yes"
    assert "server" not in response.headers


async def test_session_headers_round_trip(downstream: Downstream) -> None:
    downstream.responder = lambda _: httpx.Response(
        200,
        headers={"mcp-session-id": "session-123"},
        json={"jsonrpc": "2.0", "id": 1, "result": {}},
    )
    async with gateway_client(downstream) as gateway:
        response = await post(
            gateway, PING, **{"Mcp-Session-Id": "session-123", "MCP-Protocol-Version": "2025-11-25"}
        )

    assert response.headers["mcp-session-id"] == "session-123"
    assert downstream.last.headers["mcp-session-id"] == "session-123"
    assert downstream.last.headers["mcp-protocol-version"] == "2025-11-25"


async def test_hop_by_hop_request_headers_are_dropped(
    gateway: httpx.AsyncClient, downstream: Downstream
) -> None:
    await post(
        gateway,
        PING,
        **{
            "Connection": "keep-alive, X-Hop",
            "X-Hop": "named-by-connection",
            "Keep-Alive": "timeout=5",
            "TE": "trailers",
            "Upgrade": "h2c",
            "Proxy-Authorization": "Basic cHJveHk6c2VjcmV0",
            "X-End-To-End": "kept",
        },
    )

    forwarded = downstream.last.headers
    for name in ("x-hop", "keep-alive", "te", "upgrade", "proxy-authorization"):
        assert name not in forwarded
    assert forwarded["x-end-to-end"] == "kept"
    assert forwarded["host"] == "downstream.test"


async def test_hop_by_hop_response_headers_are_dropped(downstream: Downstream) -> None:
    downstream.responder = lambda _: httpx.Response(
        200,
        headers={"connection": "x-internal", "x-internal": "1", "keep-alive": "timeout=5", "upgrade": "h2c"},
        json={"jsonrpc": "2.0", "id": 1, "result": {}},
    )
    async with gateway_client(downstream) as gateway:
        response = await post(gateway, PING)

    for name in ("connection", "x-internal", "keep-alive", "upgrade"):
        assert name not in response.headers


async def test_downstream_is_asked_for_uncompressed_responses(
    gateway: httpx.AsyncClient, downstream: Downstream
) -> None:
    await post(gateway, PING, **{"Accept-Encoding": "gzip, br"})
    assert downstream.last.headers["accept-encoding"] == "identity"


async def test_sse_response_is_relayed(downstream: Downstream) -> None:
    downstream.responder = lambda _: httpx.Response(
        200, headers=SSE_HEADERS, stream=ScriptedStream(EVENT_1, EVENT_2)
    )
    async with gateway_client(downstream) as gateway:
        response = await post(gateway, PING, Accept="application/json, text/event-stream")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content == EVENT_1 + EVENT_2


async def test_sse_events_are_relayed_before_the_downstream_finishes(downstream: Downstream) -> None:
    release = anyio.Event()
    stream = ScriptedStream(EVENT_1, release, EVENT_2)
    downstream.responder = lambda _: httpx.Response(200, headers=SSE_HEADERS, stream=stream)
    app = make_app(downstream)
    body = json.dumps(PING).encode()
    received: list[bytes] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message["body"]:
            received.append(message["body"])
            if len(received) == 1:
                # The downstream is still holding back its second event, so nothing was buffered.
                assert not release.is_set()
                release.set()

    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    async with app.router.lifespan_context(app):
        with anyio.fail_after(5):
            await app(asgi_scope("POST", headers), receive, send)

    assert received == [EVENT_1, EVENT_2]
    assert stream.closed


async def test_client_disconnect_closes_the_downstream_stream(downstream: Downstream) -> None:
    stream = ScriptedStream(EVENT_1, anyio.Event())  # an idle SSE stream that never ends by itself
    downstream.responder = lambda _: httpx.Response(200, headers=SSE_HEADERS, stream=stream)
    app = make_app(downstream)
    first_event_sent = anyio.Event()
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await first_event_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message["body"]:
            first_event_sent.set()

    async with app.router.lifespan_context(app):
        with anyio.fail_after(5):
            await app(asgi_scope("GET", [(b"accept", b"text/event-stream")]), receive, send)

    assert stream.closed


async def test_downstream_failure_mid_stream_truncates_the_response(downstream: Downstream) -> None:
    stream = ScriptedStream(EVENT_1, httpx.ReadError("connection reset"))
    downstream.responder = lambda _: httpx.Response(200, headers=SSE_HEADERS, stream=stream)
    app = make_app(downstream)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if not sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async with app.router.lifespan_context(app):
        with anyio.fail_after(5):
            await app(asgi_scope("GET", []), receive, send)

    # Headers and the first event went out; the body is never marked complete.
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
    assert sent[1] == {"type": "http.response.body", "body": EVENT_1, "more_body": True}
    assert stream.closed


async def test_get_stream_and_delete_are_forwarded(downstream: Downstream) -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, headers=SSE_HEADERS, stream=ScriptedStream(EVENT_1))
        return httpx.Response(204)

    downstream.responder = responder
    session = {"Mcp-Session-Id": "session-123"}
    async with gateway_client(downstream) as gateway:
        stream = await gateway.get(
            "/mcp", headers={**session, "Accept": "text/event-stream", "Last-Event-ID": "7"}
        )
        deleted = await gateway.delete("/mcp", headers=session)

    assert stream.status_code == 200
    assert stream.content == EVENT_1
    assert deleted.status_code == 204
    get, delete = downstream.requests
    assert (get.method, get.content) == ("GET", b"")
    assert get.headers["last-event-id"] == "7"
    assert get.headers["mcp-session-id"] == "session-123"
    assert (delete.method, delete.headers["mcp-session-id"]) == ("DELETE", "session-123")


@pytest.mark.parametrize("method", ["PUT", "PATCH", "HEAD", "OPTIONS"])
async def test_other_http_methods_are_rejected(
    gateway: httpx.AsyncClient, downstream: Downstream, method: str
) -> None:
    response = await gateway.request(method, "/mcp")

    assert response.status_code == 405
    assert response.headers["allow"] == "GET, POST, DELETE"
    if method != "HEAD":
        assert response.json()["error"]["code"] == -32600
    assert downstream.requests == []


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (httpx.ConnectError("connection refused"), 502),
        (httpx.RemoteProtocolError("malformed status line"), 502),
        (httpx.ConnectTimeout("connect timed out"), 504),
        (httpx.ReadTimeout("read timed out"), 504),
    ],
)
async def test_downstream_failures_become_gateway_errors(
    downstream: Downstream, failure: Exception, status: int
) -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        raise failure

    downstream.responder = responder
    async with gateway_client(downstream) as gateway:
        response = await post(gateway, PING)

    assert_jsonrpc_error(response, status, GATEWAY_ERROR_CODES[status])
    assert "Traceback" not in response.text
    assert "connection refused" not in response.text  # internal detail stays in the logs


async def test_oversized_body_is_rejected_before_forwarding(downstream: Downstream) -> None:
    async with gateway_client(downstream, max_body_bytes=64) as gateway:
        response = await post(gateway, {**PING, "params": {"padding": "x" * 64}})

    assert_jsonrpc_error(response, 413, -32600)
    assert downstream.requests == []


async def test_oversized_chunked_body_is_rejected_while_streaming(downstream: Downstream) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(10):
            yield b" " * 32  # no Content-Length: the limit must hold while reading

    async with gateway_client(downstream, max_body_bytes=64) as gateway:
        response = await gateway.post("/mcp", content=chunks(), headers={"Content-Type": "application/json"})

    assert_jsonrpc_error(response, 413, -32600)
    assert downstream.requests == []


@pytest.mark.parametrize(
    "content_type",
    [
        None,
        "text/plain",
        "application/json-patch+json",
        "multipart/form-data",
        "application/json; charset=latin-1",
    ],
)
async def test_non_json_content_type_is_rejected(
    gateway: httpx.AsyncClient, downstream: Downstream, content_type: str | None
) -> None:
    headers = {} if content_type is None else {"Content-Type": content_type}
    response = await gateway.post("/mcp", content=json.dumps(PING).encode(), headers=headers)

    assert_jsonrpc_error(response, 415, -32600)
    assert downstream.requests == []


@pytest.mark.parametrize("content_type", ["application/json", "Application/JSON; charset=UTF-8"])
async def test_json_content_type_variants_are_accepted(gateway: httpx.AsyncClient, content_type: str) -> None:
    response = await gateway.post(
        "/mcp", content=json.dumps(PING).encode(), headers={"Content-Type": content_type}
    )
    assert response.status_code == 200
