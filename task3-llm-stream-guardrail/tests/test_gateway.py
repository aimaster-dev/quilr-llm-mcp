"""Gateway behaviour against a scripted upstream: streaming, errors, disconnects."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

import anyio
import httpx
import pytest
from starlette.applications import Starlette

from stream_guardrail.gateway import create_app
from stream_guardrail.redaction import REDACTED
from stream_guardrail.sse import SSEEvent, SSEParser
from tests.asgi import asgi_request
from tests.upstream import (
    SETTINGS,
    UPSTREAM_BASE,
    Handler,
    RecordingUpstream,
    completion_chunk,
    content_stream,
    split_every,
    sse,
    sse_response,
)

pytestmark = pytest.mark.anyio

R = REDACTED
PATH = "/v1/chat/completions"
STREAM_REQUEST = {"model": "mock-1", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
PLAIN_REQUEST = {"model": "mock-1", "messages": [{"role": "user", "content": "hi"}]}

MakeGateway = Callable[[Handler], tuple[Starlette, RecordingUpstream]]


@pytest.fixture
async def make_gateway() -> AsyncIterator[MakeGateway]:
    clients: list[httpx.AsyncClient] = []

    def make(handler: Handler) -> tuple[Starlette, RecordingUpstream]:
        upstream = RecordingUpstream(handler)
        client = upstream.client()
        clients.append(client)
        return create_app(SETTINGS, client), upstream

    yield make
    for client in clients:
        await client.aclose()


def events_of(body: bytes) -> list[SSEEvent]:
    parser = SSEParser()
    return parser.feed(body) + parser.close()


def content_of(events: list[SSEEvent]) -> str:
    text = []
    for event in events:
        if event.data and event.data != "[DONE]":
            for choice in json.loads(event.data).get("choices", []):
                text.append(choice["delta"].get("content") or "")
    return "".join(text)


async def test_streams_redacted_events(make_gateway: MakeGateway) -> None:
    raw = content_stream("Sure! Email ", "jane.doe@exa", "mple.com or use card 4111 ", "1111 1111 1111.")

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(split_every(raw, 7))  # cut through events, CRLFs and JSON

    app, _ = make_gateway(handler)
    async with asgi_request(app, PATH, STREAM_REQUEST) as response:
        body = await response.read()
    assert response.status == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = events_of(body)
    assert content_of(events) == f"Sure! Email {R} or use card {R}."
    assert events[-1].data == "[DONE]"
    assert b"jane" not in body
    assert b"4111" not in body


async def test_forwards_the_gateway_credential_not_the_clients(make_gateway: MakeGateway) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([content_stream("ok")])

    app, upstream = make_gateway(handler)
    headers = {"authorization": "Bearer client-key", "x-request-id": "req-123", "cookie": "s=1"}
    async with asgi_request(app, PATH, STREAM_REQUEST, headers=headers) as response:
        await response.read()
    (sent,) = upstream.requests
    assert str(sent.url) == f"{UPSTREAM_BASE}/chat/completions"
    assert sent.headers["authorization"] == "Bearer sk-gateway-secret"
    assert sent.headers["x-request-id"] == "req-123"
    assert sent.headers["accept-encoding"] == "identity"
    assert "cookie" not in sent.headers
    assert json.loads(sent.content) == STREAM_REQUEST


async def test_non_streaming_response_is_redacted(make_gateway: MakeGateway) -> None:
    completion = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "SSN 123-45-6789, mail ann@corp.io"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion)

    app, _ = make_gateway(handler)
    async with asgi_request(app, PATH, PLAIN_REQUEST) as response:
        body = await response.json()
    assert response.status == 200
    assert body["choices"][0]["message"]["content"] == f"SSN {R}, mail {R}"
    assert body["usage"] == completion["usage"]


async def test_json_reply_to_a_stream_request_is_still_redacted(make_gateway: MakeGateway) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": "a@b.io"}}]})

    app, _ = make_gateway(handler)
    async with asgi_request(app, PATH, STREAM_REQUEST) as response:
        body = await response.json()
    assert body["choices"][0]["message"]["content"] == R


@pytest.mark.parametrize(
    ("upstream_status", "status", "code"),
    [
        (400, 400, "upstream_rejected_request"),
        (404, 404, "upstream_rejected_request"),
        (401, 502, "upstream_auth_failed"),
        (403, 502, "upstream_auth_failed"),
        (429, 429, "rate_limited"),
        (500, 502, "upstream_error"),
        (503, 502, "upstream_error"),
        (504, 504, "upstream_timeout"),
    ],
)
async def test_upstream_errors_are_sanitized(
    make_gateway: MakeGateway, upstream_status: int, status: int, code: str
) -> None:
    leaky = {
        "error": {
            "message": "Invalid prompt from john@example.com with key sk-live-123",
            "type": "invalid_request_error",
            "code": "model_not_found",
        }
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(upstream_status, json=leaky, headers={"retry-after": "7"})

    app, _ = make_gateway(handler)
    for request_body in (STREAM_REQUEST, PLAIN_REQUEST):
        async with asgi_request(app, PATH, request_body) as response:
            raw = await response.read()
        assert response.status == status
        error = json.loads(raw)["error"]
        assert error["code"] == code
        assert b"john@example.com" not in raw
        assert b"sk-live" not in raw
        if status == 429:
            assert response.headers["retry-after"] == "7"
        if code == "upstream_rejected_request":
            assert error["upstream_code"] == "model_not_found"


@pytest.mark.parametrize(
    ("exception", "status", "code"),
    [
        (httpx.ConnectError, 502, "upstream_unavailable"),
        (httpx.ConnectTimeout, 504, "upstream_timeout"),
        (httpx.ReadTimeout, 504, "upstream_timeout"),
    ],
)
async def test_transport_failures_are_mapped(
    make_gateway: MakeGateway, exception: type[httpx.HTTPError], status: int, code: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise exception("internal detail: 10.0.0.7:443 refused", request=request)

    app, _ = make_gateway(handler)
    async with asgi_request(app, PATH, STREAM_REQUEST) as response:
        raw = await response.read()
    assert response.status == status
    assert json.loads(raw)["error"]["code"] == code
    assert b"10.0.0.7" not in raw


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (b"{not json", 400, "invalid_json"),
        (b"[1, 2]", 400, "invalid_request"),
        (b'{"pad": "' + b"x" * (2 << 20) + b'"}', 413, "request_too_large"),
    ],
)
async def test_bad_requests_never_reach_upstream(
    make_gateway: MakeGateway, body: bytes, status: int, code: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be called")

    app, upstream = make_gateway(handler)
    async with asgi_request(app, PATH, body) as response:
        error = (await response.json())["error"]
    assert response.status == status
    assert error["code"] == code
    assert upstream.requests == []


@pytest.mark.parametrize(
    ("extra", "code", "param"),
    [
        ({"logprobs": True}, "logprobs_not_supported", "logprobs"),
        ({"logprobs": True, "top_logprobs": 0}, "logprobs_not_supported", "logprobs"),
        ({"top_logprobs": 2}, "logprobs_not_supported", "top_logprobs"),
        ({"logprobs": 1}, "logprobs_not_supported", "logprobs"),  # truthy is enough: fail closed
        ({"modalities": ["text", "audio"], "audio": {"voice": "alloy"}}, "audio_not_supported", "modalities"),
        ({"audio": {"voice": "alloy", "format": "wav"}}, "audio_not_supported", "modalities"),
    ],
)
async def test_output_that_cannot_be_redacted_is_refused(
    make_gateway: MakeGateway, extra: dict[str, object], code: str, param: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be called")

    app, upstream = make_gateway(handler)
    for request_body in (STREAM_REQUEST, PLAIN_REQUEST):
        async with asgi_request(app, PATH, {**request_body, **extra}) as response:
            error = (await response.json())["error"]
        assert response.status == 400
        assert error["type"] == "invalid_request_error"
        assert error["code"] == code
        assert error["param"] == param
    assert upstream.requests == []


async def test_logprobs_disabled_explicitly_is_accepted(make_gateway: MakeGateway) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([content_stream("ok")])

    app, upstream = make_gateway(handler)
    request_body = {**STREAM_REQUEST, "logprobs": False, "top_logprobs": 0, "modalities": ["text"]}
    async with asgi_request(app, PATH, request_body) as response:
        await response.read()
    assert response.status == 200
    assert len(upstream.requests) == 1


def logprobs_for(text: str) -> dict[str, object]:
    token = {"token": text, "logprob": -0.1, "bytes": list(text.encode())}
    return {"content": [{**token, "top_logprobs": [token]}]}


async def test_logprobs_sent_anyway_are_dropped(make_gateway: MakeGateway) -> None:
    # A provider that ignores the missing flag (or a misconfigured one) still cannot leak tokens.
    pieces = ["Mail ", "john", "@example", ".com", " now"]
    chunks = [completion_chunk(content=p) for p in pieces]
    for c, piece in zip(chunks, pieces, strict=True):
        c["choices"][0]["logprobs"] = logprobs_for(piece)
    completion = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(pieces)},
                "logprobs": logprobs_for("john@example.com"),
                "finish_reason": "stop",
            }
        ],
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return sse_response([sse(*chunks, completion_chunk(finish="stop"), "[DONE]")])
        return httpx.Response(200, json=completion)

    app, _ = make_gateway(handler)
    async with asgi_request(app, PATH, STREAM_REQUEST) as response:
        streamed = await response.read()
    async with asgi_request(app, PATH, PLAIN_REQUEST) as response:
        plain = await response.read()
    assert content_of(events_of(streamed)) == f"Mail {R} now"
    assert json.loads(plain)["choices"][0] == {
        "index": 0,
        "message": {"role": "assistant", "content": f"Mail {R} now"},
        "logprobs": None,
        "finish_reason": "stop",
    }
    for body in (streamed, plain):
        assert b"john" not in body
        assert b"106, 111, 104, 110" not in body  # the token's UTF-8 bytes, "john"


async def test_reasoning_audio_and_content_parts_are_redacted(make_gateway: MakeGateway) -> None:
    secrets = [b"john@example.com", b"123-45-6789", b"4111 1111 1111 1111"]
    stream = sse(
        completion_chunk(role="assistant", content=""),
        completion_chunk(reasoning_content="The user's email is john@exa"),
        completion_chunk(reasoning_content="mple.com, SSN 123-45-6789."),
        completion_chunk(reasoning="Card 4111 1111 1111 1111"),
        completion_chunk(audio={"id": "audio_1", "transcript": "write to john@example.com"}),
        completion_chunk(content=[{"type": "text", "text": "john@example.com"}]),
        completion_chunk(finish="stop"),
        "[DONE]",
    )
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Card 4111 1111 1111 1111"}],
        "reasoning_content": "email john@example.com",
        "reasoning": "SSN 123-45-6789",
        "audio": {"id": "audio_1", "data": "UklGRg==", "transcript": "john@example.com"},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return sse_response(split_every(stream, 5))
        return httpx.Response(200, json={"choices": [{"index": 0, "message": message}]})

    app, _ = make_gateway(handler)
    async with asgi_request(app, PATH, STREAM_REQUEST) as response:
        streamed = await response.read()
    async with asgi_request(app, PATH, PLAIN_REQUEST) as response:
        plain = await response.json()
    for secret in secrets:
        assert secret not in streamed
    reasoning = "".join(
        choice["delta"].get("reasoning_content") or ""
        for event in events_of(streamed)
        if event.data and event.data != "[DONE]"
        for choice in json.loads(event.data)["choices"]
    )
    assert reasoning == f"The user's email is {R}, SSN {R}."
    assert plain["choices"][0]["message"] == {
        "role": "assistant",
        "content": [{"type": "text", "text": f"Card {R}"}],
        "reasoning_content": f"email {R}",
        "reasoning": f"SSN {R}",
        "audio": {"id": "audio_1", "data": "UklGRg==", "transcript": R},
    }


async def test_mid_stream_failure_flushes_then_reports_in_band(make_gateway: MakeGateway) -> None:
    async def body() -> AsyncIterator[bytes]:
        yield sse(completion_chunk(content="Contact ann@corp.io"))
        raise httpx.ReadError("connection reset by 10.0.0.7")

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(body())

    app, _ = make_gateway(handler)
    async with asgi_request(app, PATH, STREAM_REQUEST) as response:
        raw = await response.read()
    events = events_of(raw)
    assert content_of(events[:-1]) == f"Contact {R}"
    assert json.loads(events[-1].data or "")["error"]["code"] == "upstream_stream_error"
    assert all(event.data != "[DONE]" for event in events)
    assert b"10.0.0.7" not in raw


async def test_client_disconnect_cancels_the_upstream_request(make_gateway: MakeGateway) -> None:
    upstream_closed = anyio.Event()

    async def body() -> AsyncIterator[bytes]:
        try:
            yield sse(completion_chunk(content="Hello there. "))
            await anyio.sleep_forever()  # the model "thinks" forever
        finally:
            upstream_closed.set()

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(body())

    app, _ = make_gateway(handler)
    with anyio.fail_after(5):
        async with asgi_request(app, PATH, STREAM_REQUEST) as response:
            first = await response.iter_body().__anext__()
            assert b"Hello there. " in first
            response.disconnect()
            await upstream_closed.wait()
    assert not response.complete


async def test_comments_are_forwarded_as_keep_alives(make_gateway: MakeGateway) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([b": keep-alive\n\n", content_stream("hi")])

    app, _ = make_gateway(handler)
    async with asgi_request(app, PATH, STREAM_REQUEST) as response:
        raw = await response.read()
    assert raw.startswith(b": keep-alive\n\n")


async def test_healthz(make_gateway: MakeGateway) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError

    app, _ = make_gateway(handler)
    async with asgi_request(app, "/healthz", b"", method="GET") as response:
        assert await response.json() == {"status": "ok"}
