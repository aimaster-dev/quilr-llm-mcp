"""Gateway settings, validated once at startup so a misconfigured gateway refuses to start."""

from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_DOWNSTREAM_URL = "http://127.0.0.1:9000/mcp"
DEFAULT_MAX_BODY_BYTES = 1024 * 1024


class ConfigError(ValueError):
    """The gateway configuration is invalid."""


@dataclass(frozen=True, kw_only=True)
class GatewaySettings:
    downstream_url: str = DEFAULT_DOWNSTREAM_URL
    """Full URL of the downstream MCP endpoint (Streamable HTTP), e.g. http://127.0.0.1:9000/mcp."""

    connect_timeout: float = 5.0
    """Seconds to wait for a downstream connection (also used for the connection-pool wait)."""

    read_timeout: float = 60.0
    """Seconds of downstream silence tolerated on POST/DELETE; GET (SSE) streams have no read timeout."""

    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    """Largest accepted POST body. Bigger requests get 413 and never reach the downstream."""

    def __post_init__(self) -> None:
        parts = urlsplit(self.downstream_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConfigError(f"downstream URL must be an absolute http(s) URL, got {self.downstream_url!r}")
        if parts.username or parts.password:
            raise ConfigError("downstream URL must not embed credentials")
        for name in ("connect_timeout", "read_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name} must be a positive number of seconds, got {value!r}")
        if self.max_body_bytes <= 0:
            raise ConfigError(f"max_body_bytes must be positive, got {self.max_body_bytes!r}")
