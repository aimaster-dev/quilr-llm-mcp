"""The demo/test mock provider behaves as documented."""

from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette

from llm_router.mock_provider import FAKE_SECRET, INTERNAL_HOST, create_mock_app

pytestmark = pytest.mark.anyio

REQUEST = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 5}


async def post(app: Starlette, path: str = "/v1/chat/completions", **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock") as client:
        if path == "/_mock/mode":
            return await client.put(path, **kwargs)
        if path == "/_mock/stats":
            return await client.get(path)
        return await client.post(path, json=REQUEST, **kwargs)


async def test_ok_mode_returns_a_completion_with_usage() -> None:
    response = await post(create_mock_app(name="alpha"))
    body = response.json()
    assert response.status_code == 200
    assert "alpha" in body["choices"][0]["message"]["content"]
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["completion_tokens"] == 5  # the whole max_tokens budget


@pytest.mark.parametrize(("mode", "status"), [("429", 429), ("500", 500), ("400", 400)])
async def test_error_modes_include_details_a_gateway_must_hide(mode: str, status: int) -> None:
    response = await post(create_mock_app(mode=mode))
    assert response.status_code == status
    assert INTERNAL_HOST in response.text
    if mode == "429":
        assert response.headers["retry-after"] == "1"
    if mode == "500":
        assert FAKE_SECRET in response.text and "Traceback" in response.text


async def test_slow_mode_waits_before_answering() -> None:
    app = create_mock_app(mode="slow", delay=0.05)
    assert (await post(app)).status_code == 200


async def test_drip_mode_streams_a_valid_body() -> None:
    response = await post(create_mock_app(mode="drip", delay=0.05))
    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"


async def test_mode_can_be_switched_at_runtime() -> None:
    app = create_mock_app()
    assert (await post(app, "/_mock/mode", json={"mode": "429"})).json()["mode"] == "429"
    assert (await post(app)).status_code == 429
    assert (await post(app, "/_mock/mode", json={"mode": "nope"})).status_code == 400
    assert (await post(app, "/_mock/mode", content=b"[")).status_code == 400
    stats = (await post(app, "/_mock/stats")).json()
    assert stats["requests"] == 1 and stats["mode"] == "429"


def test_unknown_initial_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        create_mock_app(mode="flaky")
