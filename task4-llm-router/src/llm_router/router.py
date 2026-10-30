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

Stale keep-alive connections: a pooled connection that the provider closes as it goes idle can be
picked for a request at the same instant. Pooled connections therefore expire after 2 s, below the
usual server keep-alive (5 s for uvicorn and Node, 75 s for nginx), and a request that went out on a
*reused* connection and failed before any response arrived is sent once more on a new connection,
within the same deadline. That is the only retry: a request on a fresh connection, or one whose
response had started, is never sent twice.
"""

from __future__ import annotations

import email.utils
import json
import logging
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Annotated, Any

import anyio
import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

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
#: Idle pooled connections are dropped after this long, below common server keep-alive timeouts.
DEFAULT_KEEPALIVE_EXPIRY_SECONDS = 2.0
#: Largest provider response body read into memory; anything bigger is a bad response.
DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
#: Upper bound for a Retry-After relayed from a provider (seconds).
MAX_UPSTREAM_RETRY_AFTER_SECONDS = 60
_LOG_BODY_LIMIT = 512


class FailureKind(StrEnum):
    RATE_LIMITED = "rate_limited"  # 429
    TIMEOUT = "timeout"  # attempt deadline hit
    CONNECTION = "connection"  # connect/read/protocol errors: no usable HTTP response
    SERVER_ERROR = "server_error"  # 5xx
    BAD_RESPONSE = "bad_response"  # 2xx whose body is not a usable completion, or is too large
    REFUSED = "refused"  # other 4xx (401, 403, 404, 408, 409, 425, ...): this provider won't serve it
    CLIENT_ERROR = "client_error"  # 400/413/422: the request itself is invalid


@dataclass(frozen=True, slots=True)
class FailoverPolicy:
    """Which primary failures send the request to the secondary.

    429 and timeouts always fail over (that is the point of the router). Connection errors,
    5xx/malformed responses and refusals (a 4xx other than 400/413/422: bad provider key, unknown
    model or URL, the provider's own request timeout, conflicts) are on by default and can be
    switched off. 400/413/422 mean the request itself is invalid, so the secondary would refuse it
    too: they are never retried.
    """

    on_connection_error: bool = True
    on_server_error: bool = True
    on_refused: bool = True

    def should_failover(self, kind: FailureKind) -> bool:
        match kind:
            case FailureKind.RATE_LIMITED | FailureKind.TIMEOUT:
                return True
            case FailureKind.CONNECTION:
                return self.on_connection_error
            case FailureKind.SERVER_ERROR | FailureKind.BAD_RESPONSE:
                return self.on_server_error
            case FailureKind.REFUSED:
                return self.on_refused
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
        # A nan or infinite deadline would silently never fire (move_on_after(nan) waits forever).
        if not (math.isfinite(self.timeout) and self.timeout > 0):
            raise ValueError(f"{self.label} timeout must be a positive, finite number of seconds")

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
        retry_after: int | None = None,
    ) -> None:
        super().__init__(f"{upstream} {kind}")
        self.upstream = upstream
        self.kind = kind
        self.status = status
        self.detail = detail
        self.elapsed = elapsed
        #: Sanitized Retry-After the provider sent with a 429/503 (whole seconds, capped).
        self.retry_after = retry_after


#: Upstream 4xx statuses that mean "this request is invalid": the client's problem, not retried.
#: Any other 4xx (401/403/404 from a provider are the gateway operator's problem) is a refusal.
_REQUEST_REJECTED_STATUSES = frozenset({400, 413, 422})
#: Refusals that are transient by nature (request timeout, conflict, too early): "retry later".
_TRANSIENT_REFUSAL_STATUSES = frozenset({408, 409, 425})


def gateway_error_for(failures: list[AttemptFailure]) -> UpstreamError:
    """Map the attempt that ended routing to a sanitized gateway error.

    The client sees only a fixed message chosen here, plus a ``Retry-After`` on 503 when a
    provider sent one: the smallest of their sanitized values, i.e. when the first provider
    expects to be available again. The attempts themselves (statuses, upstream bodies, exception
    text) ride along in ``failures`` for the server logs.
    """
    final = failures[-1]
    match final.kind:
        case FailureKind.TIMEOUT:
            error_class: type[UpstreamError] = UpstreamTimeoutError
        case FailureKind.RATE_LIMITED | FailureKind.CONNECTION | FailureKind.SERVER_ERROR:
            error_class = UpstreamUnavailableError
        case FailureKind.REFUSED if final.status in _TRANSIENT_REFUSAL_STATUSES:
            error_class = UpstreamUnavailableError
        case FailureKind.CLIENT_ERROR:
            error_class = UpstreamRejectedError
        case _:
            error_class = UpstreamError
    headers: dict[str, str] = {}
    hints = [f.retry_after for f in failures if f.retry_after is not None]
    if error_class is UpstreamUnavailableError and hints:
        headers["Retry-After"] = str(min(hints))
    return error_class(failures=failures, headers=headers)


_DELTA_SECONDS = re.compile(r"\d{1,12}(?:\.\d{1,6})?", re.ASCII)


def parse_retry_after(value: str | None, *, now: float | None = None) -> int | None:
    """Sanitize an upstream ``Retry-After``: whole seconds in ``[1, 60]``, or ``None`` if unusable.

    Accepts delta-seconds (fractions are rounded up) and HTTP-dates. Only the resulting integer
    ever reaches a client, never the provider's text.
    """
    if value is None:
        return None
    value = value.strip()
    if _DELTA_SECONDS.fullmatch(value):
        seconds = float(value)
    else:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:  # HTTP-dates are always GMT; a naive result was not one
            return None
        seconds = when.timestamp() - (time.time() if now is None else now)
    return min(max(math.ceil(seconds), 1), MAX_UPSTREAM_RETRY_AFTER_SECONDS)


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


def _count_or_none(value: Any) -> int | None:
    # usage is metadata: a malformed count is dropped (and the reservation kept) rather than
    # turning a valid completion into a bad response that would be failed over.
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _mapping_or_none(value: Any) -> Any:
    return value if isinstance(value, Mapping) else None


_TokenCount = Annotated[int | None, BeforeValidator(_count_or_none)]


class _UpstreamUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt_tokens: _TokenCount = None
    completion_tokens: _TokenCount = None
    total_tokens: _TokenCount = None


class _UpstreamCompletion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    choices: list[_UpstreamChoice] = Field(min_length=1)
    usage: Annotated[_UpstreamUsage | None, BeforeValidator(_mapping_or_none)] = None


@dataclass(frozen=True, slots=True)
class Completion:
    """A successful completion, normalised to the fields the gateway returns."""

    route: str
    choices: list[dict[str, Any]]
    #: The provider's own counts, malformed fields dropped. Not yet checked for plausibility.
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


class _ExchangeProbe:
    """httpcore ``trace`` hook recording how far one HTTP/1.1 exchange got.

    ``reused`` is positive evidence that the request went out on a pooled connection: request
    headers were sent without a TCP connect in this exchange. ``responded`` means a status line and
    headers arrived. In-memory transports emit no events, so they never look reused.
    """

    def __init__(self) -> None:
        self.connected = False
        self.sent = False
        self.responded = False

    async def __call__(self, event: str, info: Mapping[str, Any]) -> None:
        if event.startswith("connection.connect_tcp."):
            self.connected = True
        elif event == "http11.send_request_headers.started":
            self.sent = True
        elif event == "http11.receive_response_headers.complete":
            self.responded = True

    @property
    def reused(self) -> bool:
        return self.sent and not self.connected

    def stale_connection(self, exc: httpx.TransportError) -> bool:
        """The provider closed a pooled connection just as it was reused, before answering."""
        stale_errors = (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)
        return self.reused and not self.responded and isinstance(exc, stale_errors)


async def _read_capped(response: httpx.Response, limit: int) -> tuple[bytes, bool]:
    """Read at most ``limit`` (decoded) bytes; the flag says the body was larger."""
    declared = response.headers.get("content-length", "")
    if declared.isascii() and declared.isdigit() and int(declared) > limit:
        return b"", True
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > limit:
            # Leaving the stream unread makes httpx drop the connection instead of pooling it.
            return bytes(body[:limit]), True
    return bytes(body), False


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
        keepalive_expiry: float = DEFAULT_KEEPALIVE_EXPIRY_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        transports: Mapping[str, httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        if secondary is not None and secondary.label == primary.label:
            raise ValueError("primary and secondary upstreams need distinct labels")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.primary = primary
        self.secondary = secondary
        self.policy = policy or FailoverPolicy()
        self.max_response_bytes = max_response_bytes
        transports = transports or {}
        self._clients = {
            upstream.label: httpx.AsyncClient(
                # A per-operation safety net only; the attempt deadline below is authoritative.
                timeout=httpx.Timeout(upstream.timeout),
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_connections // 4,
                    keepalive_expiry=keepalive_expiry,
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
        stale: httpx.TransportError | None = None  # set once a stale pooled connection was retried
        with anyio.move_on_after(upstream.timeout) as deadline:
            while True:
                probe = _ExchangeProbe()
                try:
                    async with client.stream(
                        "POST",
                        upstream.completions_url,
                        json=body,
                        headers=headers,
                        extensions={"trace": probe},
                    ) as resp:
                        raw, too_large = await _read_capped(resp, self.max_response_bytes)
                except httpx.TransportError as exc:
                    if stale is None and probe.stale_connection(exc):
                        # Nothing came back on a reused connection: the provider closed it as idle
                        # and never answered. Sent again once, on a new connection, within the
                        # same deadline.
                        logger.info(
                            "%s: pooled connection closed before any response (%r); retrying once "
                            "on a new connection",
                            upstream.label,
                            exc,
                        )
                        stale = exc
                        continue
                    kind = (
                        FailureKind.TIMEOUT
                        if isinstance(exc, httpx.TimeoutException)
                        else FailureKind.CONNECTION
                    )
                    detail = repr(exc) + (f" (after a stale-connection retry: {stale!r})" if stale else "")
                    raise _failure(upstream, kind, started, detail=detail) from None
                if too_large and 200 <= resp.status_code < 300:
                    detail = f"response body larger than {self.max_response_bytes} bytes: {_snippet(raw)}"
                    raise _failure(
                        upstream, FailureKind.BAD_RESPONSE, started, status=resp.status_code, detail=detail
                    )
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
    upstream: Upstream,
    kind: FailureKind,
    started: float,
    *,
    status: int | None = None,
    detail: str = "",
    retry_after: int | None = None,
) -> AttemptFailure:
    elapsed = anyio.current_time() - started
    return AttemptFailure(
        upstream.label, kind, status=status, detail=detail, elapsed=elapsed, retry_after=retry_after
    )


def _interpret(
    upstream: Upstream, status: int, headers: httpx.Headers, raw: bytes, started: float
) -> Completion:
    if status == 429 or status >= 500:
        kind = FailureKind.RATE_LIMITED if status == 429 else FailureKind.SERVER_ERROR
        raw_retry_after = headers.get("retry-after")
        detail = f"retry-after={raw_retry_after!r} body={_snippet(raw)}" if raw_retry_after else _snippet(raw)
        retry_after = parse_retry_after(raw_retry_after) if status in (429, 503) else None
        raise _failure(upstream, kind, started, status=status, detail=detail, retry_after=retry_after)
    if status in _REQUEST_REJECTED_STATUSES:
        raise _failure(upstream, FailureKind.CLIENT_ERROR, started, status=status, detail=_snippet(raw))
    if status >= 400:
        raise _failure(upstream, FailureKind.REFUSED, started, status=status, detail=_snippet(raw))
    if not 200 <= status < 300:
        raise _failure(upstream, FailureKind.BAD_RESPONSE, started, status=status, detail=_snippet(raw))
    try:
        parsed = _UpstreamCompletion.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        # ValueError also covers JSONDecodeError and UnicodeDecodeError.
        detail = f"{type(exc).__name__}: {_snippet(raw)}"
        raise _failure(upstream, FailureKind.BAD_RESPONSE, started, status=status, detail=detail) from None
    usage = parsed.usage.model_dump(exclude_none=True) if parsed.usage else None
    return Completion(
        route=upstream.label,
        choices=[choice.model_dump() for choice in parsed.choices],
        usage=usage or None,
    )
