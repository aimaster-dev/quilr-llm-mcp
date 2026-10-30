"""Gateway and mock provider on real uvicorn servers, talking over loopback TCP."""

from __future__ import annotations

import json
import time
from typing import Any

import anyio
import httpx
import pytest
from starlette.types import Receive, Scope, Send

from stream_guardrail.config import Settings
from stream_guardrail.gateway import create_app
from stream_guardrail.mock_provider import MockProvider, MockSettings
from stream_guardrail.redaction import REDACTED, redact
from stream_guardrail.sse import SSEEvent, SSEParser
from stream_guardrail.testing import serve

pytestmark = pytest.mark.anyio

REQUEST = {"model": "mock-llm-1", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


async def read_events(client: httpx.AsyncClient, url: str, body: dict[str, Any]) -> list[SSEEvent]:
    parser, events = SSEParser(), []
    async with client.stream("POST", url, json=body) as response:
        assert response.status_code == 200
        async for raw in response.aiter_bytes():
            events += parser.feed(raw)
    return events + parser.close()


async def test_n_choices_are_redacted_independently_end_to_end() -> None:
    # n=3 choices interleaved delta by delta, 3 characters at a time, so every pattern is split
    # across deltas while the other choices' deltas arrive in between.
    text = "Mail ann@corp.io, card 4111 1111 1111 1111 or 3782 822463 10005, SSN 078-05-1120. Bye."
    provider = MockProvider(MockSettings(text=text, chunk_size=3, delay=0, first_token_delay=0))
    with anyio.fail_after(20):
        async with serve(provider.app) as mock_url:
            gateway = create_app(Settings(upstream_base_url=f"{mock_url}/v1"))
            async with serve(gateway) as gateway_url, httpx.AsyncClient(timeout=10) as client:
                url = f"{gateway_url}/v1/chat/completions"
                events = await read_events(client, url, {**REQUEST, "n": 3})
                complete = (await client.post(url, json={**REQUEST, "stream": False, "n": 2})).json()
    texts = {0: "", 1: "", 2: ""}
    finished: list[int] = []
    for event in events[:-1]:
        for choice in json.loads(event.data or "")["choices"]:
            assert choice["index"] not in finished, "text after finish_reason"
            texts[choice["index"]] += choice["delta"].get("content") or ""
            if choice["finish_reason"] == "stop":
                finished.append(choice["index"])
    assert events[-1].data == "[DONE]"
    assert sorted(finished) == [0, 1, 2]
    expected = f"Mail {REDACTED}, card {REDACTED} or {REDACTED}, SSN {REDACTED}. Bye."
    assert redact(text) == expected
    assert texts == {0: expected, 1: expected, 2: expected}
    assert [c["message"]["content"] for c in complete["choices"]] == [expected, expected]


async def test_a_silent_upstream_mid_stream_ends_with_an_in_band_timeout() -> None:
    # The provider sends one delta, then goes quiet. httpx's read timeout applies between reads.
    hung_up = anyio.Event()
    first = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    first["choices"] = [{"index": 0, "delta": {"content": "Contact ann@corp.io"}, "finish_reason": None}]

    async def silent_upstream(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        while (await receive()).get("more_body"):
            pass
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": f"data: {json.dumps(first)}\n\n".encode(),
                "more_body": True,
            }
        )
        while (await receive())["type"] != "http.disconnect":
            pass
        hung_up.set()

    with anyio.fail_after(20):
        async with serve(silent_upstream) as upstream_url:
            gateway = create_app(Settings(upstream_base_url=f"{upstream_url}/v1", read_timeout=0.5))
            async with serve(gateway) as gateway_url, httpx.AsyncClient(timeout=10) as client:
                started = time.perf_counter()
                events = await read_events(client, f"{gateway_url}/v1/chat/completions", REQUEST)
                elapsed = time.perf_counter() - started
                with anyio.fail_after(5):
                    await hung_up.wait()  # and the gateway closed the upstream connection
    assert 0.4 < elapsed < 5
    *deltas, error = events
    content = "".join(json.loads(e.data or "")["choices"][0]["delta"]["content"] for e in deltas)
    assert content == f"Contact {REDACTED}"  # the held address was flushed, redacted
    assert json.loads(error.data or "")["error"]["code"] == "upstream_timeout"
    assert all(e.data != "[DONE]" for e in events)


async def test_streams_redacted_text_over_tcp_before_the_upstream_finishes() -> None:
    first_delta_received = anyio.Event()

    async def before_chunk(k: int) -> None:
        if k == 1:
            await first_delta_received.wait()

    provider = MockProvider(
        MockSettings(
            text="Hi! Reach me at ops@example.com or 078-05-1120.",
            chunk_size=4,
            delay=0.001,
            first_token_delay=0,
            api_key="sk-upstream",
        ),
        before_chunk=before_chunk,
    )
    parser, text = SSEParser(), ""
    with anyio.fail_after(20):
        async with serve(provider.app) as mock_url:
            gateway = create_app(Settings(upstream_base_url=f"{mock_url}/v1", upstream_api_key="sk-upstream"))
            async with serve(gateway) as gateway_url, httpx.AsyncClient(timeout=10) as client:
                url = f"{gateway_url}/v1/chat/completions"
                async with client.stream("POST", url, json=REQUEST) as response:
                    assert response.status_code == 200
                    async for raw in response.aiter_bytes():
                        for event in parser.feed(raw):
                            if event.data and event.data != "[DONE]":
                                for choice in json.loads(event.data)["choices"]:
                                    text += choice["delta"].get("content") or ""
                        if text and not first_delta_received.is_set():
                            assert text == "Hi! "
                            assert provider.stats.completed == 0
                            first_delta_received.set()
    assert text == f"Hi! Reach me at {REDACTED} or {REDACTED}."
    assert provider.stats.completed == 1


async def test_client_disconnect_cancels_the_upstream_over_tcp() -> None:
    upstream_ended = anyio.Event()
    outcomes: list[bool] = []

    def on_stream_end(completed: bool) -> None:
        outcomes.append(completed)
        upstream_ended.set()

    # ~10 s of output if nobody hangs up.
    provider = MockProvider(
        MockSettings(chunk_size=4, delay=0.05, first_token_delay=0), on_stream_end=on_stream_end
    )
    with anyio.fail_after(20):
        async with serve(provider.app) as mock_url:
            gateway = create_app(Settings(upstream_base_url=f"{mock_url}/v1"))
            async with serve(gateway) as gateway_url, httpx.AsyncClient(timeout=10) as client:
                url = f"{gateway_url}/v1/chat/completions"
                async with client.stream("POST", url, json=REQUEST) as response:
                    async for _ in response.aiter_bytes():
                        break  # read one chunk, then hang up
                with anyio.fail_after(5):
                    await upstream_ended.wait()
    assert outcomes == [False]
    assert provider.stats.cancelled == 1


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming-before-headers"])
async def test_closed_socket_cancels_the_upstream_before_it_answers(stream: bool) -> None:
    # The upstream is still "thinking", so the gateway has not sent a status line yet. A raw
    # socket that is closed is the realistic disconnect: uvicorn reports it via receive().
    started, cancelled = anyio.Event(), anyio.Event()

    async def thinking_upstream(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        while (await receive()).get("more_body"):
            pass
        started.set()
        while (await receive())["type"] != "http.disconnect":  # the gateway hung up on us
            pass
        cancelled.set()

    body = json.dumps({**REQUEST, "stream": stream}).encode()
    with anyio.fail_after(20):
        async with serve(thinking_upstream) as upstream_url:
            gateway = create_app(Settings(upstream_base_url=f"{upstream_url}/v1"))
            async with serve(gateway) as gateway_url:
                port = int(gateway_url.rsplit(":", 1)[1])
                async with await anyio.connect_tcp("127.0.0.1", port) as client:
                    await client.send(
                        b"POST /v1/chat/completions HTTP/1.1\r\nHost: gateway\r\n"
                        b"Content-Type: application/json\r\n"
                        + f"Content-Length: {len(body)}\r\n\r\n".encode()
                        + body
                    )
                    await started.wait()
                # The socket is closed. The gateway must hang up on the upstream promptly.
                with anyio.fail_after(5):
                    await cancelled.wait()
