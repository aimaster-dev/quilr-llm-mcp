"""Black-box runs over real TCP: uvicorn serves the gateway and two mock providers on ephemeral ports.

These exercise what in-process transports cannot: real connection setup and teardown, and the
primary noticing that the gateway hung up on it after the deadline.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import httpx
import pytest
import uvicorn
from starlette.applications import Starlette

from llm_router.app import create_app
from llm_router.mock_provider import FAKE_SECRET, INTERNAL_HOST, create_mock_app
from tests.gateway import KEY_A, chat_body, make_settings

pytestmark = pytest.mark.anyio

DEADLINE = 0.3


@asynccontextmanager
async def serve(app: Starlette) -> AsyncIterator[str]:
    """Run ``app`` under uvicorn on 127.0.0.1 with an OS-assigned port; yield its base URL."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_config=None, log_level="warning", timeout_graceful_shutdown=2)
    server = uvicorn.Server(config)
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(server.serve, [sock])
            with anyio.fail_after(10):
                while not server.started:
                    await anyio.sleep(0.01)
            try:
                yield f"http://127.0.0.1:{port}"
            finally:
                server.should_exit = True
    finally:
        sock.close()


@asynccontextmanager
async def stack(tmp_path: Path, primary_mode: str, **settings: object) -> AsyncIterator[tuple]:
    primary_app = create_mock_app(name="primary-mock", mode=primary_mode, delay=5.0)
    secondary_app = create_mock_app(name="secondary-mock")
    async with serve(primary_app) as primary_url, serve(secondary_app) as secondary_url:
        gateway_settings = make_settings(
            tmp_path,
            primary_url=f"{primary_url}/v1",
            secondary_url=f"{secondary_url}/v1",
            primary_timeout=DEADLINE,
            **settings,
        )
        async with (
            serve(create_app(gateway_settings)) as gateway_url,
            httpx.AsyncClient(base_url=gateway_url, trust_env=False, timeout=10) as client,
        ):
            yield client, primary_app.state.mock, secondary_app.state.mock, (primary_url, secondary_url)


async def wait_for(predicate, timeout: float = 3.0) -> None:
    with anyio.fail_after(timeout):
        while not predicate():
            await anyio.sleep(0.02)


def auth(key: str = KEY_A) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.mark.parametrize("primary_mode", ["slow", "drip"])
async def test_hanging_primary_fails_over_at_the_deadline_and_is_hung_up_on(
    tmp_path: Path, primary_mode: str
) -> None:
    async with stack(tmp_path, primary_mode) as (client, primary, secondary, _):
        started = anyio.current_time()
        response = await client.post("/v1/chat/completions", json=chat_body(), headers=auth())
        elapsed = anyio.current_time() - started

        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "fallback"
        assert "secondary-mock" in response.json()["choices"][0]["message"]["content"]
        assert DEADLINE - 0.05 <= elapsed < DEADLINE + 1.5
        # The primary sees the gateway close the connection: nothing was left dangling.
        await wait_for(lambda: primary.disconnects == 1)
        assert secondary.requests == 1


async def test_walkthrough_failover_errors_and_rate_limits(tmp_path: Path) -> None:
    async with stack(tmp_path, "ok", tokens_per_window=1_000) as (client, _, _, urls):
        primary_url, secondary_url = urls

        ok = await client.post("/v1/chat/completions", json=chat_body(), headers=auth())
        assert ok.status_code == 200 and ok.headers["x-gateway-route"] == "primary"

        await client.put(f"{primary_url}/_mock/mode", json={"mode": "429"})
        failover = await client.post("/v1/chat/completions", json=chat_body(), headers=auth())
        assert failover.status_code == 200 and failover.headers["x-gateway-route"] == "fallback"

        await client.put(f"{secondary_url}/_mock/mode", json={"mode": "500"})
        down = await client.post("/v1/chat/completions", json=chat_body(), headers=auth())
        assert down.status_code == 503
        assert down.json()["error"]["code"] == "upstream_unavailable"
        for leak in (INTERNAL_HOST, FAKE_SECRET, "Traceback", "primary-mock", "secondary-mock", "127.0.0.1"):
            assert leak not in down.text

        await client.put(f"{primary_url}/_mock/mode", json={"mode": "ok"})
        statuses = []
        for _ in range(60):
            response = await client.post("/v1/chat/completions", json=chat_body(), headers=auth())
            statuses.append(response.status_code)
            if response.status_code == 429:
                break
        assert statuses[-1] == 429
        assert int(response.headers["retry-after"]) >= 1
        assert response.json()["error"]["code"] == "tenant_token_limit_exceeded"
        assert int(response.headers["x-ratelimit-remaining"]) < 100
