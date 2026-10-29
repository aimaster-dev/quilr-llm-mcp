"""Harness that runs the real gateway app (with its lifespan) against scripted upstreams."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette

from llm_router.app import Gateway, create_app
from llm_router.keys import fingerprint_api_key
from llm_router.settings import Settings
from tests.upstreams import Recorder

KEY_A = "tenant-a-RAW-api-key-000111"
KEY_B = "tenant-b-RAW-api-key-000222"
PROMPT = "x" * 40  # 10 tokens; with message overhead (4) and reply priming (3) the prompt is 17
COST = 100  # 17 + max_tokens 83


def chat_body(max_tokens: int | None = COST - 17, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "gpt-test", "messages": [{"role": "user", "content": PROMPT}], **extra}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "primary_url": "https://primary.provider.test/v1",
        "primary_model": "primary-model",
        "primary_api_key": "sk-provider-primary",
        "primary_timeout": 0.25,
        "secondary_url": "https://fallback.provider.test/v1",
        "secondary_model": "backup-model",
        "secondary_api_key": "sk-provider-backup",
        "secondary_timeout": 1.0,
        "tenant_key_fingerprints": frozenset({fingerprint_api_key(KEY_A), fingerprint_api_key(KEY_B)}),
        "db_path": tmp_path / "gateway.db",
        "tokens_per_window": 10_000,
        "sweep_interval": 60.0,
    }
    values.update(overrides)
    return Settings(**values)


@dataclass
class Harness:
    app: Starlette
    client: httpx.AsyncClient

    @property
    def gateway(self) -> Gateway:
        return self.app.state.gateway

    async def chat(self, key: str | None = KEY_A, body: Any = None, **kwargs: Any) -> httpx.Response:
        headers = kwargs.pop("headers", {})
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        return await self.client.post(
            "/v1/chat/completions", json=chat_body() if body is None else body, headers=headers, **kwargs
        )

    async def usage(self, key: str = KEY_A) -> int:
        return await self.gateway.limiter.usage(fingerprint_api_key(key))


@asynccontextmanager
async def running_gateway(
    settings: Settings,
    primary: Recorder,
    secondary: Recorder | None = None,
    *,
    clock: Callable[[], float] = time.time,
) -> AsyncIterator[Harness]:
    transports = {"primary": primary.transport}
    if secondary is not None:
        transports["fallback"] = secondary.transport
    app = create_app(settings, transports=transports, clock=clock)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
            yield Harness(app, client)
