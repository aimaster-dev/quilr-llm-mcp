"""Gateway behaviour against a scripted upstream: streaming, errors, disconnects."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

import anyio
import httpx
import pytest
from starlette.applications import Starlette

from stream_guardrail.completions import ChatStreamRedactor
from stream_guardrail.config import Settings
from stream_guardrail.gateway import Gateway, create_app
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


async def test_events_in_the_same_read_as_an_oversized_one_are_delivered() -> None:
    good = sse(completion_chunk(role="assistant", content=""), completion_chunk(content="Shipped to ann@ex"))
    oversized = b"data: " + b"x" * 2000  # no line break: the event can only grow

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([good + oversized])  # one network read carries all of it

    settings = Settings(upstream_base_url=UPSTREAM_BASE, max_event_size=1000)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        app = create_app(settings, client)
        async with asgi_request(app, PATH, STREAM_REQUEST) as response:
            raw = await response.read()
    events = events_of(raw)
    # The good events, then the held "ann@ex" flushed as end-of-text, then the in-band error.
    assert content_of(events[:-1]) == "Shipped to ann@ex"
    assert json.loads(events[-1].data or "")["error"]["code"] == "upstream_invalid_stream"
    assert b"xxxx" not in raw


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


@pytest.mark.parametrize("request_body", [PLAIN_REQUEST, STREAM_REQUEST], ids=["non-streaming", "streaming"])
async def test_client_disconnect_before_the_upstream_answers_cancels_it(
    make_gateway: MakeGateway, request_body: dict[str, object]
) -> None:
    # The model is still thinking: no status line has been sent to the client yet.
    upstream_started, upstream_cancelled = anyio.Event(), anyio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_started.set()
        try:
            await anyio.sleep_forever()
        finally:
            upstream_cancelled.set()
        raise AssertionError("unreachable")

    app, _ = make_gateway(handler)
    with anyio.fail_after(5):
        async with asgi_request(app, PATH, request_body, wait_for_start=False) as response:
            await upstream_started.wait()
            response.disconnect()
            await upstream_cancelled.wait()
            await response.app_done.wait()
    assert response.status == 0  # nothing was sent


async def test_client_disconnect_while_reading_a_complete_answer_cancels_it(
    make_gateway: MakeGateway,
) -> None:
    body_started, upstream_closed = anyio.Event(), anyio.Event()

    async def slow_body() -> AsyncIterator[bytes]:
        try:
            yield b'{"choices": [{"index": 0, "message": {"content": "'
            body_started.set()
            await anyio.sleep_forever()
        finally:
            upstream_closed.set()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=slow_body())

    app, _ = make_gateway(handler)
    with anyio.fail_after(5):
        async with asgi_request(app, PATH, PLAIN_REQUEST, wait_for_start=False) as response:
            await body_started.wait()
            response.disconnect()
            await upstream_closed.wait()
            await response.app_done.wait()


async def test_lone_surrogates_from_upstream_are_escaped(make_gateway: MakeGateway) -> None:
    # "\ud83d" is valid JSON but not encodable as UTF-8; the gateway used to crash on it.
    chunk = b'{"choices":[{"index":0,"delta":{"content":"Hi \\ud83d a@b.io!"},"finish_reason":null}]}'
    complete = b'{"choices":[{"index":0,"message":{"content":"Hi \\ud83d a@b.io!"}}]}'

    async def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return sse_response([b"data: " + chunk + b"\n\n" + sse("[DONE]")])
        return httpx.Response(200, headers={"content-type": "application/json"}, content=complete)

    app, _ = make_gateway(handler)
    async with asgi_request(app, PATH, STREAM_REQUEST) as response:
        streamed = await response.read()
    async with asgi_request(app, PATH, PLAIN_REQUEST) as response:
        plain = await response.read()
    assert response.status == 200
    assert response.headers["content-type"] == "application/json"
    assert content_of(events_of(streamed)) == f"Hi \ud83d {R}!"
    assert events_of(streamed)[-1].data == "[DONE]"
    assert json.loads(plain)["choices"][0]["message"]["content"] == f"Hi \ud83d {R}!"
    assert streamed.isascii()
    assert plain.isascii()


async def test_deeply_nested_json_is_handled(make_gateway: MakeGateway) -> None:
    deep = b"[" * 100_000 + b"]" * 100_000

    async def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return sse_response([b"data: " + deep + b"\n\n", content_stream("still here")])
        return httpx.Response(200, headers={"content-type": "application/json"}, content=deep)

    app, upstream = make_gateway(handler)
    # Request: refused before the upstream is called.
    async with asgi_request(app, PATH, b'{"messages": ' + deep + b"}") as response:
        error = (await response.json())["error"]
    assert (response.status, error["code"]) == (400, "invalid_json")
    assert upstream.requests == []
    # Complete answer: a sanitized 502.
    async with asgi_request(app, PATH, PLAIN_REQUEST) as response:
        error = (await response.json())["error"]
    assert (response.status, error["code"]) == (502, "upstream_invalid_response")
    # Streamed event: redacted as plain text, and the stream goes on.
    async with asgi_request(app, PATH, STREAM_REQUEST) as response:
        events = events_of(await response.read())
    assert content_of([e for e in events if e.data and e.data.startswith("{")]) == "still here"
    assert events[-1].data == "[DONE]"


class ExplodingRedactor(ChatStreamRedactor):
    """Fails on the third event with an exception whose text must never reach the client."""

    def __init__(self) -> None:
        super().__init__()
        self.events = 0

    def process(self, event: SSEEvent) -> list[SSEEvent]:
        self.events += 1
        if self.events == 3:
            raise RuntimeError("internal detail: secret-token-123")
        return super().process(event)


async def test_unexpected_relay_error_flushes_then_reports_in_band() -> None:
    raw = content_stream("Write to ann@ex", "ample.com", "more")

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([raw])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        app = create_app(SETTINGS, client, redactor_factory=ExplodingRedactor)
        async with asgi_request(app, PATH, STREAM_REQUEST) as response:
            body = await response.read()
    assert response.complete  # a clean end of the chunked body, not a broken connection
    events = events_of(body)
    error = json.loads(events[-1].data or "")["error"]
    assert error == {
        "message": "The gateway failed while relaying the response.",
        "type": "server_error",
        "param": None,
        "code": "internal_error",
    }
    # "ann@ex" was held when the failure hit; it is flushed (as end-of-text), never raw "ann@example".
    assert content_of(events[:-1]) == "Write to ann@ex"
    assert b"example" not in body
    assert b"secret-token" not in body
    assert all(event.data != "[DONE]" for event in events)


async def test_unexpected_error_before_streaming_is_a_sanitized_500(make_gateway: MakeGateway) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("internal detail: secret-token-123")  # not an httpx error

    app, _ = make_gateway(handler)
    for request_body in (STREAM_REQUEST, PLAIN_REQUEST):
        async with asgi_request(app, PATH, request_body) as response:
            raw = await response.read()
        assert response.status == 500
        assert json.loads(raw)["error"]["code"] == "internal_error"
        assert b"secret-token" not in raw


async def test_app_level_errors_use_the_openai_shape(
    make_gateway: MakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode(self: object, request: object) -> bytes:
        raise RuntimeError("internal detail: secret-token-123")

    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be called")

    monkeypatch.setattr(Gateway, "_read_request", explode)
    app, _ = make_gateway(handler)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        response = await client.post(PATH, json=PLAIN_REQUEST)
    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "error": {
            "message": "The gateway failed to process the request.",
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        }
    }


@pytest.mark.parametrize(
    ("upstream_status", "headers", "code"),
    [
        (301, {"location": "https://internal.example/v1"}, "upstream_error"),
        (302, {"location": "https://internal.example/v1"}, "upstream_error"),
        (307, {"location": "https://internal.example/v1"}, "upstream_error"),
        (304, {}, "upstream_error"),
        (204, {}, "upstream_invalid_response"),  # a success without a body
    ],
)
async def test_non_2xx_and_bodyless_answers_are_sanitized(
    make_gateway: MakeGateway, upstream_status: int, headers: dict[str, str], code: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        content = b"" if upstream_status in (204, 304) else b'{"redirect": "john@example.com"}'
        return httpx.Response(
            upstream_status, headers={"content-type": "application/json", **headers}, content=content
        )

    app, _ = make_gateway(handler)
    for request_body in (STREAM_REQUEST, PLAIN_REQUEST):
        async with asgi_request(app, PATH, request_body) as response:
            raw = await response.read()
        assert response.status == 502
        assert "location" not in response.headers
        assert json.loads(raw)["error"]["code"] == code
        assert b"john" not in raw
        assert b"internal.example" not in raw


@pytest.mark.parametrize("content_type", ["text/html", "text/plain", "application/json"])
async def test_a_2xx_answer_that_is_not_json_is_sanitized(
    make_gateway: MakeGateway, content_type: str
) -> None:
    # A proxy's error page, or an SSE body without the event-stream content type.
    page = b'<html>Bad gateway for john@example.com</html>\ndata: {"choices": []}\n\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": content_type}, content=page)

    app, _ = make_gateway(handler)
    for request_body in (STREAM_REQUEST, PLAIN_REQUEST):
        async with asgi_request(app, PATH, request_body) as response:
            raw = await response.read()
        assert response.status == 502
        assert json.loads(raw)["error"] == {
            "message": "The upstream response is not valid JSON.",
            "type": "upstream_error",
            "param": None,
            "code": "upstream_invalid_response",
        }
        assert b"john" not in raw


async def test_a_complete_answer_over_the_size_limit_is_refused() -> None:
    answer = {"choices": [{"index": 0, "message": {"content": "x" * 50_000 + " john@example.com"}}]}
    chunks_sent = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal chunks_sent
        raw = json.dumps(answer).encode()
        for i in range(0, len(raw), 1000):
            chunks_sent += 1
            yield raw[i : i + 1000]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body())

    settings = Settings(upstream_base_url=UPSTREAM_BASE, max_response_bytes=4096)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        app = create_app(settings, client)
        async with asgi_request(app, PATH, PLAIN_REQUEST) as response:
            raw = await response.read()
    assert response.status == 502
    assert json.loads(raw)["error"]["code"] == "upstream_response_too_large"
    assert b"john" not in raw
    assert chunks_sent == 5  # reading stopped as soon as the limit was crossed (of 51 chunks)


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
