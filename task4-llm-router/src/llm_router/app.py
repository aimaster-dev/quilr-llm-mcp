"""The gateway HTTP API: ``POST /v1/chat/completions`` (OpenAI-compatible, non-streaming).

Request flow::

    authenticate (Bearer key -> tenant fingerprint)
    -> read + validate body          (400 / 413)
    -> estimate cost, reserve tokens (413 too large, 429 rate limited)
    -> route: primary, maybe secondary
    -> success: reconcile the reservation with the provider's usage (if plausible)
       failure: release the reservation, return a sanitized error

Quota policy: a tenant is charged for the completion it receives, measured by the provider's
``usage`` when a correct provider could have reported it (``tokens.accepted_usage``), otherwise by
the reservation itself. A completion is always returned once it exists, even if the bookkeeping
fails. When no completion is returned (both providers failed, the request was rejected, the
client went away) the reservation is released. While a request is in flight its reservation counts
in full, so a tenant can never have more than its budget outstanding.

Client disconnects: ASGI servers such as uvicorn do not cancel a handler whose client hung up; they
only report ``http.disconnect`` on the receive channel. Routing therefore runs next to a watcher
that waits for that message and cancels the upstream attempt, so a client that has gone away is
neither failed over nor charged.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
import anyio.to_thread
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from llm_router.errors import (
    AuthenticationError,
    GatewayError,
    InvalidRequestError,
    RateLimitError,
    RequestTooLargeError,
    ServiceUnavailableError,
)
from llm_router.keys import TenantKeyring
from llm_router.limiter import (
    AsyncTokenRateLimiter,
    LimitDecision,
    LimiterUnavailableError,
    Outcome,
)
from llm_router.logging_config import configure_logging
from llm_router.middleware import EXCEPTION_HANDLERS, MIDDLEWARE, current_request_id
from llm_router.router import Completion, ModelRouter
from llm_router.schemas import parse_chat_request
from llm_router.settings import Settings, load_settings
from llm_router.tokens import accepted_usage, estimate_prompt_tokens, max_plausible_usage, request_cost

logger = logging.getLogger(__name__)

ROUTE_HEADER = "X-Gateway-Route"


@dataclass(frozen=True)
class Gateway:
    """Long-lived resources shared by all requests (created in the app lifespan)."""

    settings: Settings
    keyring: TenantKeyring
    limiter: AsyncTokenRateLimiter
    router: ModelRouter


def _rate_limit_headers(limit: int, remaining: int) -> dict[str, str]:
    return {"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": str(max(remaining, 0))}


async def _read_body(request: Request, limit: int) -> bytes:
    too_large = RequestTooLargeError(f"The request body exceeds {limit} bytes.", code="payload_too_large")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise too_large
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


async def _release(gateway: Gateway, tenant: str, decision: LimitDecision) -> int | None:
    """Give back a reservation; returns the tenant's remaining tokens when known."""
    assert decision.reservation_id is not None
    try:
        await gateway.limiter.release(decision.reservation_id)
        used = await gateway.limiter.usage(tenant)
    except (sqlite3.Error, LimiterUnavailableError):
        # Fail safe: the row keeps its estimate until the window drains.
        logger.exception("could not release reservation %s", decision.reservation_id)
        return None
    return decision.limit - used


class _NoResponse(Response):
    """Sends nothing: used when the client has already hung up, so there is nobody to answer."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        return None


async def _route_unless_disconnected(
    request: Request, router: ModelRouter, payload: dict[str, Any], request_id: str
) -> Completion | None:
    """Run ``router.complete`` while watching for the client to hang up.

    Returns ``None`` if the client disconnected first: the upstream attempt has then been
    cancelled (closing its connection) and no failover was started. Router errors propagate as-is,
    and so does cancellation from outside (server shutdown).
    """
    disconnected = False
    completion: Completion | None = None
    failure: Exception | None = None

    async def watch_for_disconnect(scope: anyio.CancelScope) -> None:
        nonlocal disconnected
        # The body has been read in full, so the only message the server can still deliver is
        # http.disconnect, once the connection is lost (or after the response has been sent).
        while (await request.receive())["type"] != "http.disconnect":
            pass
        disconnected = True
        scope.cancel()

    async with anyio.create_task_group() as tg:
        tg.start_soon(watch_for_disconnect, tg.cancel_scope)
        try:
            completion = await router.complete(payload, request_id=request_id)
        except Exception as exc:
            # Kept out of the task group, which would wrap it in an ExceptionGroup.
            failure = exc
        # Routing is over (or was cancelled by the watcher): stop watching.
        tg.cancel_scope.cancel()

    if disconnected:
        return None
    if failure is not None:
        raise failure
    assert completion is not None
    return completion


async def chat_completions(request: Request) -> Response:
    gateway: Gateway = request.app.state.gateway
    settings = gateway.settings
    request_id = current_request_id(request)

    tenant = gateway.keyring.authenticate_header(request.headers.get("authorization"))
    if tenant is None:
        raise AuthenticationError()

    chat = parse_chat_request(await _read_body(request, settings.max_body_bytes))
    if chat.stream:
        raise InvalidRequestError(
            'Streaming is not supported by this gateway; send "stream": false.',
            code="streaming_not_supported",
        )

    max_tokens = chat.max_tokens or settings.default_max_tokens
    prompt_tokens = estimate_prompt_tokens(chat.messages_payload())
    cost = request_cost(prompt_tokens, max_tokens)

    # Shielded: once the reservation transaction has run we must learn its id, or a cancelled
    # request would leave tokens reserved until the window drains.
    with anyio.CancelScope(shield=True):
        try:
            decision = await gateway.limiter.reserve(tenant, cost)
        except (sqlite3.Error, LimiterUnavailableError) as exc:
            raise ServiceUnavailableError(
                code="rate_limiter_unavailable", log_detail=f"limiter database error: {exc!r}"
            ) from None

    headers = _rate_limit_headers(decision.limit, decision.remaining)
    log_tenant = tenant[:12]
    if decision.outcome is Outcome.TOO_LARGE:
        raise RequestTooLargeError(
            f"This request needs an estimated {cost} tokens (prompt plus max_tokens), more than the "
            f"{decision.limit}-token window allows. Reduce max_tokens or the prompt.",
            headers=headers,
            log_detail=f"tenant={log_tenant} cost={cost}",
        )
    if decision.outcome is Outcome.RATE_LIMITED:
        assert decision.retry_after_header is not None
        raise RateLimitError(
            decision.retry_after_header,
            headers=headers,
            log_detail=(
                f"tenant={log_tenant} used={decision.used} cost={cost} "
                f"retry_after={decision.retry_after:.3f}s"
            ),
        )
    assert decision.reservation_id is not None

    started = anyio.current_time()
    try:
        completion = await _route_unless_disconnected(
            request, gateway.router, chat.upstream_payload(max_tokens), request_id
        )
    except BaseException as exc:
        # No completion reaches the client, so nothing is charged.
        with anyio.CancelScope(shield=True):
            remaining = await _release(gateway, tenant, decision)
        if isinstance(exc, GatewayError) and remaining is not None:
            exc.headers.update(_rate_limit_headers(decision.limit, remaining))
        raise
    if completion is None:
        with anyio.CancelScope(shield=True):
            await _release(gateway, tenant, decision)
        logger.info(
            "client disconnected after %.0f ms: upstream attempt cancelled, no failover, "
            "reservation of %d tokens released (tenant=%s)",
            (anyio.current_time() - started) * 1000,
            cost,
            log_tenant,
        )
        return _NoResponse()

    # From here on a completion exists: whatever happens to the bookkeeping, the client gets it.
    reported = completion.total_tokens
    charged = accepted_usage(reported, prompt_tokens=prompt_tokens, max_tokens=max_tokens)
    if charged is None and reported is not None:
        logger.warning(
            "ignoring implausible provider usage total=%d (possible for this request: 1..%d); "
            "the %d-token reservation is kept",
            reported,
            max_plausible_usage(prompt_tokens, max_tokens),
            cost,
        )
    remaining: int | None = None
    if charged is not None and charged != cost:
        with anyio.CancelScope(shield=True):
            try:
                remaining = await gateway.limiter.reconcile(decision.reservation_id, charged)
            except Exception:
                # Fail safe: the reservation keeps its (upper-bound) estimate.
                logger.exception(
                    "could not reconcile reservation %s; it keeps its %d-token estimate",
                    decision.reservation_id,
                    cost,
                )
    logger.info(
        "completion served route=%s tenant=%s estimate=%d reported=%s charged=%d failed_attempts=%d",
        completion.route,
        log_tenant,
        cost,
        reported,
        cost if charged is None else charged,
        len(completion.failed_attempts),
    )

    body: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": chat.model or "default",
        "choices": completion.choices,
    }
    if charged is not None and completion.usage is not None:
        # Relayed only when plausible: the gateway does not pass on numbers it refused to charge.
        body["usage"] = completion.usage
    headers = _rate_limit_headers(decision.limit, decision.remaining if remaining is None else remaining)
    headers[ROUTE_HEADER] = completion.route
    return JSONResponse(body, headers=headers)


async def healthz(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


def create_app(
    settings: Settings,
    *,
    transports: Mapping[str, httpx.AsyncBaseTransport] | None = None,
    clock: Callable[[], float] = time.time,
) -> Starlette:
    """Build the gateway. ``transports`` and ``clock`` are injection points for tests."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        sync_limiter = await anyio.to_thread.run_sync(settings.open_limiter, clock)
        limiter = AsyncTokenRateLimiter(sync_limiter, fail_fast_seconds=settings.db_fail_fast)
        try:
            async with (
                ModelRouter(
                    settings.primary_upstream(),
                    settings.secondary_upstream(),
                    policy=settings.failover_policy(),
                    max_response_bytes=settings.max_response_bytes,
                    transports=transports,
                ) as router,
                anyio.create_task_group() as tg,
            ):
                app.state.gateway = Gateway(
                    settings, TenantKeyring(settings.tenant_key_fingerprints), limiter, router
                )
                tg.start_soon(limiter.run_sweeper, settings.sweep_interval)
                logger.info(
                    "gateway ready: %d tenant key(s), %d tokens per %gs window, primary deadline %.0f ms, "
                    "fallback %s",
                    len(settings.tenant_key_fingerprints),
                    settings.tokens_per_window,
                    settings.window_seconds,
                    settings.primary_timeout * 1000,
                    "configured" if settings.secondary_upstream() else "not configured",
                )
                yield
                tg.cancel_scope.cancel()
        finally:
            limiter.close()

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/healthz", healthz, methods=["GET"]),
        ],
        middleware=MIDDLEWARE,
        exception_handlers=EXCEPTION_HANDLERS,
        lifespan=lifespan,
    )


def create_app_from_env() -> Starlette:
    """App factory for multi-worker servers: every worker rebuilds the settings from the environment."""
    settings = load_settings([], os.environ)
    configure_logging(settings.log_level)
    return create_app(settings)
