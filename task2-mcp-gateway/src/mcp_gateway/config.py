"""Gateway settings, validated once at startup so a misconfigured gateway refuses to start."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import urlsplit

DEFAULT_DOWNSTREAM_URL = "http://127.0.0.1:9000/mcp"
DEFAULT_MAX_BODY_BYTES = 1024 * 1024
DEFAULT_MAX_STREAMS_PER_SUBJECT = 10
# The name of the environment variable that holds the secret, not the secret itself.
JWT_SECRET_ENV = "MCP_GATEWAY_JWT_SECRET"  # noqa: S105
DEFAULT_JWT_ISSUER = "mcp-gateway-dev"
DEFAULT_JWT_AUDIENCE = "mcp-gateway"
# RFC 7518 section 3.2: an HS256 key must be at least as long as the hash output (256 bits).
MIN_JWT_SECRET_BYTES = 32
MAX_JWT_LEEWAY_SECONDS = 300


class ConfigError(ValueError):
    """The gateway configuration is invalid."""


def check_jwt_secret(secret: str) -> None:
    if len(secret.encode()) < MIN_JWT_SECRET_BYTES:
        raise ConfigError(f"${JWT_SECRET_ENV} must hold a secret of at least {MIN_JWT_SECRET_BYTES} bytes")


@dataclass(frozen=True, kw_only=True)
class GatewaySettings:
    jwt_secret: str = field(repr=False)
    """Shared HS256 secret used to verify bearer tokens. Never logged (excluded from repr)."""

    jwt_issuer: str = DEFAULT_JWT_ISSUER
    """Required `iss` claim."""

    jwt_audience: str = DEFAULT_JWT_AUDIENCE
    """Required `aud` claim: tokens minted for another service are rejected."""

    jwt_leeway: float = 30.0
    """Clock skew, in seconds, tolerated when checking `exp`, `nbf` and `iat`."""

    downstream_url: str = DEFAULT_DOWNSTREAM_URL
    """Full URL of the downstream MCP endpoint (Streamable HTTP), e.g. http://127.0.0.1:9000/mcp."""

    connect_timeout: float = 5.0
    """Seconds to wait for a downstream connection (also used for the connection-pool wait)."""

    read_timeout: float = 60.0
    """Seconds of downstream silence tolerated on POST/DELETE; GET (SSE) streams have no read timeout."""

    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    """Largest accepted POST body. Bigger requests get 413 and never reach the downstream."""

    max_streams_per_subject: int = DEFAULT_MAX_STREAMS_PER_SUBJECT
    """Concurrent GET (server-to-client SSE) streams one token subject may hold open; then 429."""

    def __post_init__(self) -> None:
        check_jwt_secret(self.jwt_secret)
        if not self.jwt_issuer or not self.jwt_audience:
            raise ConfigError("the JWT issuer and audience must not be empty")
        if not 0 <= self.jwt_leeway <= MAX_JWT_LEEWAY_SECONDS:
            raise ConfigError(f"jwt_leeway must be between 0 and {MAX_JWT_LEEWAY_SECONDS} seconds")
        parts = urlsplit(self.downstream_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConfigError("downstream URL must be an absolute http(s) URL")
        if parts.username or parts.password:
            raise ConfigError("downstream URL must not embed credentials")
        for name in ("connect_timeout", "read_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name} must be a positive number of seconds, got {value!r}")
        for name in ("max_body_bytes", "max_streams_per_subject"):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(f"{name} must be positive, got {value!r}")
