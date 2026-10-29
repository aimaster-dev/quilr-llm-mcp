"""Gateway settings, read from ``GUARDRAIL_*`` environment variables (CLI flags override them)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

ENV_PREFIX = "GUARDRAIL_"


@dataclass(frozen=True, slots=True)
class Settings:
    upstream_base_url: str = "http://127.0.0.1:9000/v1"
    # The gateway's own credential for the provider. Clients never send it and never see it, and
    # the client's Authorization header is not forwarded. Read from the environment only, so it
    # does not show up in process listings.
    upstream_api_key: str | None = field(default=None, repr=False)
    connect_timeout: float = 5.0
    read_timeout: float = 60.0  # longest silence allowed between two upstream reads
    write_timeout: float = 10.0
    pool_timeout: float = 5.0
    max_request_bytes: int = 1 << 20
    max_response_bytes: int = 8 << 20  # non-streaming responses only; streams are never buffered
    max_event_size: int = 1 << 20  # one SSE event

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **overrides: object) -> Settings:
        env = os.environ if environ is None else environ
        values: dict[str, object] = {}
        for name, kind in (
            ("upstream_base_url", str),
            ("upstream_api_key", str),
            ("connect_timeout", float),
            ("read_timeout", float),
            ("write_timeout", float),
            ("pool_timeout", float),
            ("max_request_bytes", int),
            ("max_response_bytes", int),
            ("max_event_size", int),
        ):
            raw = env.get(ENV_PREFIX + name.upper())
            if raw:
                values[name] = kind(raw)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)  # type: ignore[arg-type]

    @property
    def chat_completions_url(self) -> str:
        return self.upstream_base_url.rstrip("/") + "/chat/completions"
