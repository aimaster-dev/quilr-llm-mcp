"""Configuration from command-line flags and ``LLM_ROUTER_*`` environment variables.

Flags win over environment variables, which win over defaults. Secrets (provider API keys and
tenant keys) are read from the environment only, so they never appear in process listings.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_router.keys import TenantKeyring, fingerprint_api_key
from llm_router.limiter import DEFAULT_TOKENS_PER_WINDOW, DEFAULT_WINDOW_SECONDS
from llm_router.router import (
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_PRIMARY_TIMEOUT_SECONDS,
    DEFAULT_SECONDARY_TIMEOUT_SECONDS,
    FailoverPolicy,
    Upstream,
)
from llm_router.schemas import MAX_TOKENS_LIMIT

ENV_PREFIX = "LLM_ROUTER_"
PRIMARY_LABEL = "primary"
SECONDARY_LABEL = "fallback"


@dataclass(frozen=True)
class Settings:
    primary_url: str
    primary_model: str
    #: SHA-256 fingerprints of the tenant API keys allowed to call the gateway.
    tenant_key_fingerprints: frozenset[str]
    primary_api_key: str | None = field(default=None, repr=False)
    primary_timeout: float = DEFAULT_PRIMARY_TIMEOUT_SECONDS
    secondary_url: str | None = None
    secondary_model: str | None = None
    secondary_api_key: str | None = field(default=None, repr=False)
    secondary_timeout: float = DEFAULT_SECONDARY_TIMEOUT_SECONDS
    failover_on_connection_error: bool = True
    failover_on_server_error: bool = True
    failover_on_refused: bool = True
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    db_path: Path = Path("data/llm_router.db")
    tokens_per_window: int = DEFAULT_TOKENS_PER_WINDOW
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    sweep_interval: float = 30.0
    default_max_tokens: int = 1024
    max_body_bytes: int = 1 << 20
    host: str = "127.0.0.1"
    port: int = 8080
    workers: int = 1
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.primary_url or not self.primary_model:
            raise ValueError("a primary upstream URL and model are required")
        if bool(self.secondary_url) != bool(self.secondary_model):
            raise ValueError("set both the secondary upstream URL and model, or neither")
        for name in ("primary_timeout", "secondary_timeout", "window_seconds", "sweep_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tokens_per_window < 1:
            raise ValueError("tokens_per_window must be at least 1")
        if not 1 <= self.default_max_tokens <= MAX_TOKENS_LIMIT:
            raise ValueError(f"default_max_tokens must be between 1 and {MAX_TOKENS_LIMIT}")
        if self.max_body_bytes < 1024:
            raise ValueError("max_body_bytes must be at least 1024")
        if self.max_response_bytes < 1024:
            raise ValueError("max_response_bytes must be at least 1024")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        TenantKeyring(self.tenant_key_fingerprints)  # validates the fingerprints

    def primary_upstream(self) -> Upstream:
        return Upstream(
            PRIMARY_LABEL, self.primary_url, self.primary_model, self.primary_timeout, self.primary_api_key
        )

    def secondary_upstream(self) -> Upstream | None:
        if not self.secondary_url or not self.secondary_model:
            return None
        return Upstream(
            SECONDARY_LABEL,
            self.secondary_url,
            self.secondary_model,
            self.secondary_timeout,
            self.secondary_api_key,
        )

    def failover_policy(self) -> FailoverPolicy:
        return FailoverPolicy(
            on_connection_error=self.failover_on_connection_error,
            on_server_error=self.failover_on_server_error,
            on_refused=self.failover_on_refused,
        )

    def to_env(self) -> dict[str, str]:
        """Environment that reproduces these settings (handed to multi-worker server processes)."""
        env: dict[str, str] = {}
        for option in _OPTIONS:
            value = getattr(self, option.attr)
            if value is None:
                continue
            env[option.env] = option.to_env(value)
        env[ENV_PREFIX + "TENANT_KEY_SHA256"] = ",".join(sorted(self.tenant_key_fingerprints))
        if self.primary_api_key:
            env[ENV_PREFIX + "PRIMARY_API_KEY"] = self.primary_api_key
        if self.secondary_api_key:
            env[ENV_PREFIX + "SECONDARY_API_KEY"] = self.secondary_api_key
        return env


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def _ms_to_seconds(value: str) -> float:
    return float(value) / 1000


@dataclass(frozen=True)
class _Option:
    flag: str
    attr: str
    parse: Callable[[str], Any]
    help: str
    to_env: Callable[[Any], str] = str
    boolean: bool = False

    @property
    def env(self) -> str:
        return ENV_PREFIX + self.flag.removeprefix("--").replace("-", "_").upper()


def _seconds_to_ms(value: float) -> str:
    return f"{value * 1000:g}"


_OPTIONS = (
    _Option("--host", "host", str, "interface to bind (default 127.0.0.1)"),
    _Option("--port", "port", int, "port to listen on (default 8080)"),
    _Option("--workers", "workers", int, "server worker processes sharing the limiter database (default 1)"),
    _Option("--db-path", "db_path", Path, "SQLite database file (default data/llm_router.db)"),
    _Option("--tokens-per-window", "tokens_per_window", int, "token budget per tenant (default 50000)"),
    _Option("--window-seconds", "window_seconds", float, "sliding window length (default 60)"),
    _Option("--sweep-interval", "sweep_interval", float, "seconds between expired-row sweeps (default 30)"),
    _Option(
        "--default-max-tokens",
        "default_max_tokens",
        int,
        "max_tokens when the client omits it (default 1024)",
    ),
    _Option("--max-body-bytes", "max_body_bytes", int, "largest accepted request body (default 1 MiB)"),
    _Option("--primary-url", "primary_url", str, "primary OpenAI-compatible base URL, e.g. https://host/v1"),
    _Option("--primary-model", "primary_model", str, "model name sent to the primary"),
    _Option(
        "--primary-timeout-ms",
        "primary_timeout",
        _ms_to_seconds,
        "hard deadline for the whole primary attempt (default 3000)",
        to_env=_seconds_to_ms,
    ),
    _Option("--secondary-url", "secondary_url", str, "secondary (fallback) base URL"),
    _Option("--secondary-model", "secondary_model", str, "model name sent to the secondary"),
    _Option(
        "--secondary-timeout-ms",
        "secondary_timeout",
        _ms_to_seconds,
        "deadline for the secondary attempt (default 30000)",
        to_env=_seconds_to_ms,
    ),
    _Option(
        "--failover-on-connection-error",
        "failover_on_connection_error",
        _parse_bool,
        "fail over when the primary cannot be reached (default on)",
        to_env=lambda v: "true" if v else "false",
        boolean=True,
    ),
    _Option(
        "--failover-on-server-error",
        "failover_on_server_error",
        _parse_bool,
        "fail over on primary 5xx or malformed responses (default on)",
        to_env=lambda v: "true" if v else "false",
        boolean=True,
    ),
    _Option(
        "--failover-on-refused",
        "failover_on_refused",
        _parse_bool,
        "fail over when the primary refuses with a 4xx other than 400/413/422/429, e.g. 401, 403, "
        "404, 408 (default on)",
        to_env=lambda v: "true" if v else "false",
        boolean=True,
    ),
    _Option(
        "--max-response-bytes",
        "max_response_bytes",
        int,
        "largest provider response body read; bigger is a bad response (default 4 MiB)",
    ),
    _Option("--log-level", "log_level", str, "DEBUG, INFO, WARNING or ERROR (default INFO)"),
)


def build_parser(prog: str = "llm-router") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "LLM gateway router with a per-tenant token rate limiter and primary/secondary failover. "
            f"Every flag can also be set with an {ENV_PREFIX}* environment variable. Secrets are read "
            f"from the environment only: {ENV_PREFIX}TENANT_KEYS (comma-separated tenant keys) or "
            f"{ENV_PREFIX}TENANT_KEY_SHA256 (their SHA-256 fingerprints), {ENV_PREFIX}PRIMARY_API_KEY "
            f"and {ENV_PREFIX}SECONDARY_API_KEY."
        ),
    )
    for option in _OPTIONS:
        help_text = f"{option.help} [env: {option.env}]"
        if option.boolean:
            parser.add_argument(
                option.flag, dest=option.attr, action=argparse.BooleanOptionalAction, help=help_text
            )
        else:
            metavar = option.flag.removeprefix("--").replace("-", "_").upper()
            parser.add_argument(
                option.flag, dest=option.attr, type=option.parse, metavar=metavar, help=help_text
            )
    return parser


def _split(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def load_settings(argv: Sequence[str] | None, environ: Mapping[str, str]) -> Settings:
    """Build settings from flags and environment; exits with a usage error when invalid."""
    parser = build_parser()
    args = parser.parse_args(argv)
    values: dict[str, Any] = {}
    for option in _OPTIONS:
        value = getattr(args, option.attr)
        if value is None and environ.get(option.env):
            try:
                value = option.parse(environ[option.env])
            except (ValueError, argparse.ArgumentTypeError) as exc:
                parser.error(f"{option.env}: {exc}")
        if value is not None:
            values[option.attr] = value

    fingerprints = {fingerprint_api_key(key) for key in _split(environ.get(ENV_PREFIX + "TENANT_KEYS"))}
    fingerprints |= {fp.lower() for fp in _split(environ.get(ENV_PREFIX + "TENANT_KEY_SHA256"))}
    if not fingerprints:
        parser.error(
            f"no tenant keys configured: set {ENV_PREFIX}TENANT_KEYS or {ENV_PREFIX}TENANT_KEY_SHA256"
        )
    if "primary_url" not in values or "primary_model" not in values:
        parser.error("--primary-url and --primary-model are required")

    try:
        return Settings(
            tenant_key_fingerprints=frozenset(fingerprints),
            primary_api_key=environ.get(ENV_PREFIX + "PRIMARY_API_KEY") or None,
            secondary_api_key=environ.get(ENV_PREFIX + "SECONDARY_API_KEY") or None,
            **values,
        )
    except ValueError as exc:
        parser.error(str(exc))
