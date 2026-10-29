"""Gateway and mock provider on real uvicorn servers, talking over loopback TCP."""

from __future__ import annotations

import json

import anyio
import httpx
import pytest

from stream_guardrail.config import Settings
from stream_guardrail.gateway import create_app
from stream_guardrail.mock_provider import MockProvider, MockSettings
from stream_guardrail.redaction import REDACTED
from stream_guardrail.sse import SSEParser
from stream_guardrail.testing import serve

pytestmark = pytest.mark.anyio

REQUEST = {"model": "mock-llm-1", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


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
