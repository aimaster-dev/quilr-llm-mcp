"""The mock provider itself: chunking, options, and the streaming transport used to reach it."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from stream_guardrail.mock_provider import MockProvider, MockSettings
from stream_guardrail.sse import SSEParser
from stream_guardrail.testing import StreamingASGITransport

pytestmark = pytest.mark.anyio

FAST = MockSettings(text="Hello, world!", chunk_size=4, delay=0, first_token_delay=0)


@pytest.fixture
async def provider() -> AsyncIterator[MockProvider]:
    yield MockProvider(FAST)


async def post(provider: MockProvider, body: dict[str, Any], **headers: str) -> httpx.Response:
    async with httpx.AsyncClient(transport=StreamingASGITransport(provider.app)) as client:
        response = await client.post("http://mock/v1/chat/completions", json=body, headers=headers)
        await response.aread()
        return response


def chunks_of(raw: bytes) -> list[Any]:
    parser = SSEParser()
    events = parser.feed(raw) + parser.close()
    return [e.data if e.data == "[DONE]" else json.loads(e.data or "") for e in events]


async def test_streams_text_in_fixed_size_deltas(provider: MockProvider) -> None:
    response = await post(provider, {"stream": True, "messages": []})
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks = chunks_of(response.content)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    pieces = [c["choices"][0]["delta"]["content"] for c in chunks[1:-2]]
    assert pieces == ["Hell", "o, w", "orld", "!"]
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1] == "[DONE]"
    assert provider.stats.completed == 1


async def test_n_choices_usage_and_overrides(provider: MockProvider) -> None:
    body = {
        "stream": True,
        "n": 2,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "echo me"}],
        "mock": {"chunk_size": 100, "echo": True, "repeat": 2},
    }
    chunks = chunks_of((await post(provider, body)).content)
    texts = {0: "", 1: ""}
    for chunk in chunks[:-1]:
        for choice in chunk["choices"]:
            texts[choice["index"]] += choice["delta"].get("content", "")
    assert texts == {0: "echo meecho me", 1: "echo meecho me"}
    assert chunks[-2]["usage"]["completion_tokens"] > 0


async def test_non_streaming_completion(provider: MockProvider) -> None:
    body = (await post(provider, {"messages": []})).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello, world!"


@pytest.mark.parametrize(
    "mock", [{"chunk_size": 0}, {"delay_ms": -1}, {"repeat": "3"}, {"chunk_size": True}, {"text": 5}]
)
async def test_rejects_bad_options(provider: MockProvider, mock: dict[str, Any]) -> None:
    response = await post(provider, {"stream": True, "mock": mock})
    assert response.status_code == 400


async def test_requires_the_api_key_when_configured() -> None:
    provider = MockProvider(MockSettings(api_key="sk-mock", first_token_delay=0))
    assert (await post(provider, {"messages": []})).status_code == 401
    ok = await post(provider, {"messages": []}, authorization="Bearer sk-mock")
    assert ok.status_code == 200
