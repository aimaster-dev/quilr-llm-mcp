"""A misconfigured gateway refuses to start instead of failing on the first request."""

from __future__ import annotations

from typing import Any

import pytest

from mcp_gateway.config import ConfigError, GatewaySettings


@pytest.mark.parametrize(
    "overrides",
    [
        {"downstream_url": "ftp://downstream.test/mcp"},
        {"downstream_url": "/mcp"},
        {"downstream_url": "http:///mcp"},
        {"downstream_url": "http://user:secret@downstream.test/mcp"},
        {"connect_timeout": 0},
        {"read_timeout": -1},
        {"read_timeout": float("inf")},
        {"max_body_bytes": 0},
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        GatewaySettings(**overrides)


def test_defaults_are_valid() -> None:
    settings = GatewaySettings()
    assert settings.downstream_url == "http://127.0.0.1:9000/mcp"
    assert settings.max_body_bytes == 1024 * 1024
