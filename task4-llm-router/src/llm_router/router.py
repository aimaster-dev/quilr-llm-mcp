"""Model router: a primary provider with a hard deadline, and failover to a secondary provider.

Every attempt runs under its own deadline (``anyio.move_on_after``) that covers the *whole*
exchange: waiting for a pooled connection, connecting, sending, receiving headers, reading the
body and parsing it. Per-operation socket timeouts are not enough on their own: a provider that
drips one byte every second never trips a read timeout but can take minutes to finish.

When the deadline fires, the in-flight request is cancelled. Leaving the ``client.stream()`` block
closes the response, and httpx drops a connection whose body was not fully read instead of
returning it to the pool. Attempts are strictly sequential, so by the time the secondary is called
the primary exchange is already gone: a late primary response cannot be used, logged as a success
or counted against the tenant.

Cancellation from *outside* the router (client disconnect, shutdown) is not a timeout and is never
turned into a failover; it propagates unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import anyio
import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_router.errors import (
    UpstreamError,
    UpstreamRejectedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)

logger = logging.getLogger(__name__)

DEFAULT_PRIMARY_TIMEOUT_SECONDS = 3.0
DEFAULT_SECONDARY_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_CONNECTIONS = 200
_LOG_BODY_LIMIT = 512


class FailureKind(StrEnum):
    RATE_LIMITED = "rate_limited"  # 429
    TIMEOUT = "timeout"  # attempt deadline hit
    CONNECTION = "connection"  # connect/read/protocol errors: no usable HTTP response
    SERVER_ERROR = "server_error"  # 5xx
    BAD_RESPONSE = "bad_response"  # 2xx whose body is not a usable completion
    CLIENT_ERROR = "client_error"  # any other 4xx: the request itself was refused


@dataclass(frozen=True, slots=True)
class FailoverPolicy:
    """Which primary failures send the request to the secondary.

    429 and timeouts always fail over (that is the point of the router). Connection errors and
    5xx/malformed responses are on by default and can be switched off. Other 4xx responses mean the
    request itself is bad, so the secondary would refuse it too: they are never retried.
    """

    on_connection_error: bool = True
    on_server_error: bool = True

    def should_failover(self, kind: FailureKind) -> bool:
        match kind:
            case FailureKind.RATE_LIMITED | FailureKind.TIMEOUT:
                return True
            case FailureKind.CONNECTION:
                return self.on_connection_error
            case FailureKind.SERVER_ERROR | FailureKind.BAD_RESPONSE:
                return self.on_server_error
            case _:
                return False


@dataclass(frozen=True, slots=True)
class Upstream:
    """An OpenAI-compatible chat-completions provider."""

    label: str  # "primary" / "fallback": opaque, safe to expose
    base_url: str
    model: str
    timeout: float
    api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError(f"{self.label} timeout must be positive")

    @property
    def completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


class AttemptFailure(Exception):
    """One failed call to one upstream. ``detail`` is for server logs only."""

    def __init__(
        self,
        upstream: str,
        kind: FailureKind,
        *,
        status: int | None = None,
        detail: str = "",
        elapsed: float = 0.0,
    ) -> None:
        super().__init__(f"{upstream} {kind}")
        self.upstream = upstream
        self.kind = kind
        self.status = status
        self.detail = detail
        self.elapsed = elapsed


#: Upstream 4xx statuses that mean "this request is invalid" rather than "the provider is broken or
#: misconfigured" (401/403/404 from a provider are the gateway operator's problem, not the client's).
_REQUEST_REJECTED_STATUSES = frozenset({400, 413, 422})


def gateway_error_for(failures: list[AttemptFailure]) -> UpstreamError:
    """Map the attempt that ended routing to a sanitized gateway error.

    The client sees only a fixed message chosen here. The attempts themselves (statuses, upstream
    bodies, exception text) ride along in ``failures`` for the server logs.
    """
    final = failures[-1]
    match final.kind:
        case FailureKind.TIMEOUT:
            error_class: type[UpstreamError] = UpstreamTimeoutError
        case FailureKind.RATE_LIMITED | FailureKind.CONNECTION | FailureKind.SERVER_ERROR:
            error_class = UpstreamUnavailableError
        case FailureKind.CLIENT_ERROR if final.status in _REQUEST_REJECTED_STATUSES:
            error_class = UpstreamRejectedError
        case _:
            error_class = UpstreamError
    return error_class(failures=failures)


# -- upstream response schema --------------------------------------------------------------------
# Only these fields are read from a provider. Anything else it sends (provider-specific fields,
# fingerprints, debugging data) is dropped and never reaches the client.


class _UpstreamMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str = "assistant"
    content: str | None = None


class _UpstreamChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: int = 0
    message: _UpstreamMessage
    finish_reason: str | None = None


class _UpstreamUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class _UpstreamCompletion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    choices: list[_UpstreamChoice] = Field(min_length=1)
    usage: _UpstreamUsage | None = None


@dataclass(frozen=True, slots=True)
class Completion:
    """A successful completion, normalised to the fields the gateway returns."""

    route: str
    choices: list[dict[str, Any]]
    usage: dict[str, int] | None
    #: Prior failed attempts (e.g. the primary that timed out), for logging.
    failed_attempts: tuple[AttemptFailure, ...] = ()

    @property
    def total_tokens(self) -> int | None:
        if self.usage is None:
            return None
        if "total_tokens" in self.usage:
            return self.usage["total_tokens"]
        if "prompt_tokens" in self.usage and "completion_tokens" in self.usage:
            return self.usage["prompt_tokens"] + self.usage["completion_tokens"]
        return None


def _snippet(raw: bytes) -> str:
    """A bounded, escaped excerpt of an upstream body for the logs.

    repr() escapes newlines and control characters, so an upstream cannot forge extra log lines.
    """
    text = raw[:_LOG_BODY_LIMIT].decode("utf-8", errors="replace")
    return repr(text) + ("..." if len(raw) > _LOG_BODY_LIMIT else "")


class ModelRouter:
    """Routes chat completions to the primary upstream and fails over to the secondary.

    Each upstream gets its own connection pool, so a primary that hangs and exhausts its pool can
    never starve requests that are failing over. Safe to share across concurrent requests.
    """

    def __init__(
        self,
        primary: Upstream,
        secondary: Upstream | None = None,
        *,
        policy: FailoverPolicy | None = None,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        transports: Mapping[str, httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        if secondary is not None and secondary.label == primary.label:
            raise ValueError("primary and secondary upstreams need distinct labels")
        self.primary = primary
        self.secondary = secondary
        self.policy = policy or FailoverPolicy()
        transports = transports or {}
        self._clients = {
            upstream.label: httpx.AsyncClient(
                # A per-operation safety net only; the attempt deadline below is authoritative.
                timeout=httpx.Timeout(upstream.timeout),
                limits=httpx.Limits(
                    max_connections=max_connections, max_keepalive_connections=max_connections // 4
                ),
                transport=transports.get(upstream.label),
                follow_redirects=False,
                # Egress is explicit configuration: ambient proxy variables must not be able to
                # route provider traffic (and provider API keys) somewhere else.
                trust_env=False,
            )
            for upstream in (primary, secondary)
            if upstream is not None
        }

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()

    async def __aenter__(self) -> ModelRouter:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def complete(self, request: Mapping[str, Any], *, request_id: str | None = None) -> Completion:
        """Run ``request`` (OpenAI chat-completions fields, minus ``model``) with failover.

        Raises an ``UpstreamError`` subclass when no upstream produced a completion.
        """
        failures: list[AttemptFailure] = []
        candidates = [self.primary] if self.secondary is None else [self.primary, self.secondary]
        for upstream in candidates:
            try:
                completion = await self._attempt(upstream, request, request_id)
            except AttemptFailure as failure:
                failures.append(failure)
                logger.warning(
                    "%s attempt failed: kind=%s status=%s elapsed_ms=%.0f detail=%s",
                    failure.upstream,
                    failure.kind,
                    failure.status,
                    failure.elapsed * 1000,
                    failure.detail,
                )
                if not self.policy.should_failover(failure.kind):
                    break
                continue
            return replace(completion, failed_attempts=tuple(failures))
        raise gateway_error_for(failures)

    async def _attempt(
        self, upstream: Upstream, request: Mapping[str, Any], request_id: str | None
    ) -> Completion:
        client = self._clients[upstream.label]
        body = {**request, "model": upstream.model, "stream": False}
        headers = {"Accept": "application/json"}
        if upstream.api_key:
            headers["Authorization"] = f"Bearer {upstream.api_key}"
        if request_id:
            headers["X-Request-Id"] = request_id

        started = anyio.current_time()
        with anyio.move_on_after(upstream.timeout) as deadline:
            try:
                async with client.stream(
                    "POST", upstream.completions_url, json=body, headers=headers
                ) as resp:
                    raw = await resp.aread()
            except httpx.TimeoutException as exc:
                raise _failure(upstream, FailureKind.TIMEOUT, started, detail=repr(exc)) from None
            except httpx.TransportError as exc:
                raise _failure(upstream, FailureKind.CONNECTION, started, detail=repr(exc)) from None
            # Parsing is part of the attempt: the result is only used if this whole block finished.
            return _interpret(upstream, resp.status_code, resp.headers, raw, started)

        # Only reachable when *this* deadline cancelled the block; outer cancellation propagates.
        assert deadline.cancelled_caught
        raise _failure(
            upstream,
            FailureKind.TIMEOUT,
            started,
            detail=f"no complete response within {upstream.timeout * 1000:.0f} ms",
        )


def _failure(
    upstream: Upstream, kind: FailureKind, started: float, *, status: int | None = None, detail: str = ""
) -> AttemptFailure:
    elapsed = anyio.current_time() - started
    return AttemptFailure(upstream.label, kind, status=status, detail=detail, elapsed=elapsed)


def _interpret(
    upstream: Upstream, status: int, headers: httpx.Headers, raw: bytes, started: float
) -> Completion:
    if status == 429:
        detail = f"retry-after={headers.get('retry-after')!r} body={_snippet(raw)}"
        raise _failure(upstream, FailureKind.RATE_LIMITED, started, status=status, detail=detail)
    if status >= 500:
        raise _failure(upstream, FailureKind.SERVER_ERROR, started, status=status, detail=_snippet(raw))
    if status >= 400:
        raise _failure(upstream, FailureKind.CLIENT_ERROR, started, status=status, detail=_snippet(raw))
    if not 200 <= status < 300:
        raise _failure(upstream, FailureKind.BAD_RESPONSE, started, status=status, detail=_snippet(raw))
    try:
        parsed = _UpstreamCompletion.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        # ValueError also covers JSONDecodeError and UnicodeDecodeError.
        detail = f"{type(exc).__name__}: {_snippet(raw)}"
        raise _failure(upstream, FailureKind.BAD_RESPONSE, started, status=status, detail=detail) from None
    return Completion(
        route=upstream.label,
        choices=[choice.model_dump() for choice in parsed.choices],
        usage=parsed.usage.model_dump(exclude_none=True) if parsed.usage else None,
    )
