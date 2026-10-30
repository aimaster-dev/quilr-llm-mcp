"""Latency and memory properties of the gateway relay, end to end through the mock provider.

The upstream is the mock provider reached through ``StreamingASGITransport``. The client side is
driven through the raw ASGI interface, so the tests see each write the moment the gateway makes it.
"""

from __future__ import annotations

import itertools
import json
import tracemalloc
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import anyio
import anyio.lowlevel
import httpx
import pytest
from starlette.applications import Starlette

from stream_guardrail.completions import ChatStreamRedactor
from stream_guardrail.config import Settings
from stream_guardrail.gateway import create_app
from stream_guardrail.mock_provider import DEFAULT_TEXT, MockProvider, MockSettings
from stream_guardrail.redaction import MAX_BUFFERED, REDACTED, redact
from stream_guardrail.sse import SSEEvent, SSEParser
from stream_guardrail.testing import StreamingASGITransport
from tests.asgi import asgi_request

pytestmark = pytest.mark.anyio

PATH = "/v1/chat/completions"
REQUEST = {"model": "mock-llm-1", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


@asynccontextmanager
async def gateway_over(
    provider: MockProvider, redactor_factory: Callable[[], ChatStreamRedactor] = ChatStreamRedactor
) -> AsyncIterator[Starlette]:
    async with httpx.AsyncClient(transport=StreamingASGITransport(provider.app)) as upstream:
        yield create_app(Settings(upstream_base_url="http://mock/v1"), upstream, redactor_factory)


class ContentReader:
    """Incrementally decodes the gateway's SSE output into the concatenated content."""

    def __init__(self) -> None:
        self.parser = SSEParser()
        self.text = ""
        self.done = False

    def feed(self, raw: bytes) -> None:
        for event in self.parser.feed(raw):
            if event.data == "[DONE]":
                self.done = True
            elif event.data:
                for choice in json.loads(event.data).get("choices", []):
                    self.text += choice["delta"].get("content") or ""


async def test_first_safe_chunk_reaches_the_client_before_the_upstream_finishes() -> None:
    client_saw_first_text = anyio.Event()

    async def before_chunk(k: int) -> None:
        if k == 1:  # the upstream refuses to go on until the client has the first delta
            await client_saw_first_text.wait()

    text = "Hello, your card 4111 1111 1111 1111 is on file. Bye."
    provider = MockProvider(
        MockSettings(text=text, chunk_size=7, delay=0, first_token_delay=0), before_chunk=before_chunk
    )
    reader = ContentReader()
    with anyio.fail_after(10):  # a buffering gateway would deadlock here
        async with gateway_over(provider) as app, asgi_request(app, PATH, REQUEST) as response:
            async for raw in response.iter_body():
                reader.feed(raw)
                if not client_saw_first_text.is_set() and reader.text:
                    assert reader.text == "Hello, "  # the first delta, forwarded on its own
                    assert provider.stats.completed == 0
                    client_saw_first_text.set()
    assert reader.done
    assert reader.text == f"Hello, your card {REDACTED} is on file. Bye."
    assert provider.stats.completed == 1


async def test_text_held_back_waits_only_for_the_next_delta() -> None:
    # "Sure" could still become "Sure@x.io", so it waits for one more delta, and no longer.
    released = anyio.Event()
    seen: list[str] = []

    async def before_chunk(k: int) -> None:
        if k == 2:
            await released.wait()

    provider = MockProvider(
        MockSettings(text="Sure! Done.", chunk_size=4, delay=0, first_token_delay=0),
        before_chunk=before_chunk,
    )
    reader = ContentReader()
    with anyio.fail_after(10):
        async with gateway_over(provider) as app, asgi_request(app, PATH, REQUEST) as response:
            async for raw in response.iter_body():
                reader.feed(raw)
                if reader.text and not released.is_set():
                    seen.append(reader.text)
                    released.set()
    assert seen == ["Sure! "]  # upstream sent "Sure" and "! Do"; the client already has "Sure! "
    assert reader.text == "Sure! Done."


async def test_each_event_of_a_burst_is_handed_over_before_the_next_is_redacted() -> None:
    # One upstream read carries a whole answer. A real server's send() returns without yielding,
    # so unless the relay lets the event loop run after each write, the server (asyncio's
    # Proactor loop on Windows, for one) puts nothing but the first write on the wire until the
    # whole burst has been redacted. Observe that with a task that counts loop iterations.
    burst = b"".join(
        b"data: "
        + json.dumps({"choices": [{"index": 0, "delta": {"content": f"word{i} "}}]}).encode()
        + b"\n\n"
        for i in range(50)
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=burst)

    ticks = 0
    seen_at: list[int] = []
    body = json.dumps(REQUEST).encode()
    sent_request = False

    async def receive() -> dict[str, object]:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    async def send(message: dict[str, object]) -> None:  # like uvicorn: no await, no yield
        if message["type"] == "http.response.body" and message.get("body"):
            seen_at.append(ticks)

    async def tick() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await anyio.lowlevel.checkpoint()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": PATH,
        "raw_path": PATH.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        app = create_app(Settings(upstream_base_url="http://mock/v1"), upstream)
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tg:
                tg.start_soon(tick)
                await app(scope, receive, send)  # type: ignore[arg-type]
                tg.cancel_scope.cancel()
    assert len(seen_at) >= 50
    assert all(b > a for a, b in itertools.pairwise(seen_at)), seen_at


class RecordingRedactor(ChatStreamRedactor):
    """Records how many characters the gateway's redactor holds after every event."""

    def __init__(self) -> None:
        super().__init__()
        self.peak = 0
        self.events = 0

    def process(self, event: SSEEvent) -> list[SSEEvent]:
        out = super().process(event)
        self.events += 1
        self.peak = max(self.peak, self.buffered)
        return out


async def test_redactor_buffer_stays_bounded_on_a_very_long_stream() -> None:
    redactors: list[RecordingRedactor] = []

    def factory() -> RecordingRedactor:
        redactors.append(RecordingRedactor())
        return redactors[-1]

    repeat = 300  # ~80k characters in ~5,000 deltas
    provider = MockProvider(MockSettings(chunk_size=16, delay=0, first_token_delay=0, repeat=repeat))
    reader = ContentReader()
    with anyio.fail_after(60):
        async with gateway_over(provider, factory) as app, asgi_request(app, PATH, REQUEST) as response:
            async for raw in response.iter_body():
                reader.feed(raw)
    (redactor,) = redactors
    assert reader.done
    assert reader.text == redact(DEFAULT_TEXT) * repeat
    assert redactor.events > 5_000
    assert 0 < redactor.peak <= MAX_BUFFERED
    assert redactor.peak < 64  # in practice: the tail of one word or one number


async def test_memory_stays_flat_while_streaming_a_megabyte() -> None:
    # Peak traced memory is ~100 KB for this 1 MB stream and stays there at 4 MB. A gateway that
    # accumulated the response would need at least the response size.
    text = DEFAULT_TEXT * 16  # ~4 KB, streamed 250 times
    provider = MockProvider(
        MockSettings(text=text, chunk_size=2048, delay=0, first_token_delay=0, repeat=250)
    )
    received = 0
    tracemalloc.start()
    try:
        with anyio.fail_after(60):
            async with gateway_over(provider) as app, asgi_request(app, PATH, REQUEST) as response:
                tracemalloc.reset_peak()
                async for raw in response.iter_body():
                    received += len(raw)  # counted, not kept
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert received > 1_000_000
    assert peak < 400_000, f"peak traced memory {peak} bytes for a {received}-byte stream"
