"""Model router: deadline enforcement, failover decisions, cancellation and concurrency."""

from __future__ import annotations

import json
import logging
import random

import anyio
import anyio.abc
import httpx
import pytest

from llm_router.errors import (
    UpstreamError,
    UpstreamRejectedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from llm_router.router import (
    DEFAULT_KEEPALIVE_EXPIRY_SECONDS,
    DEFAULT_PRIMARY_TIMEOUT_SECONDS,
    Completion,
    FailoverPolicy,
    FailureKind,
    ModelRouter,
    Upstream,
)
from tests.upstreams import (
    DripStream,
    Recorder,
    completion_body,
    connection_refused,
    drip,
    ok,
    respond,
    scripted_server,
)

pytestmark = pytest.mark.anyio

DEADLINE = 0.25  # the production default is 3.0 s; a short one keeps the suite fast
SECONDARY_TIMEOUT = 0.75
REQUEST = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}


def make_router(
    primary: Recorder, secondary: Recorder | None, policy: FailoverPolicy | None = None
) -> ModelRouter:
    transports = {"primary": primary.transport}
    fallback = None
    if secondary is not None:
        transports["fallback"] = secondary.transport
        fallback = Upstream(
            "fallback", "https://backup.example/v1", "backup-model", SECONDARY_TIMEOUT, api_key="sk-backup"
        )
    return ModelRouter(
        Upstream("primary", "https://primary.example/v1", "primary-model", DEADLINE, api_key="sk-primary"),
        fallback,
        policy=policy,
        transports=transports,
    )


async def run(router: ModelRouter, request: dict | None = None) -> Completion:
    async with router:
        return await router.complete(request or REQUEST, request_id="req_test")


async def fail(router: ModelRouter) -> UpstreamError:
    with pytest.raises(UpstreamError) as excinfo:
        await run(router)
    return excinfo.value


def test_default_primary_deadline_is_3000_ms() -> None:
    assert DEFAULT_PRIMARY_TIMEOUT_SECONDS == 3.0


def test_upstream_rejects_non_positive_timeouts() -> None:
    with pytest.raises(ValueError):
        Upstream("primary", "http://x", "m", 0)


def test_upstream_labels_must_differ() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ModelRouter(Upstream("primary", "http://a", "m", 1), Upstream("primary", "http://b", "m", 1))


# -- happy path ----------------------------------------------------------------------------------


async def test_healthy_primary_never_touches_the_secondary() -> None:
    primary, secondary = Recorder(ok("from primary")), Recorder(ok("from secondary"))
    completion = await run(make_router(primary, secondary))

    assert completion.route == "primary"
    assert completion.choices[0]["message"]["content"] == "from primary"
    assert completion.total_tokens == 42
    assert completion.failed_attempts == ()
    assert secondary.requests == []

    sent = primary.json()
    assert sent["model"] == "primary-model"
    assert sent["stream"] is False
    assert sent["messages"] == REQUEST["messages"]
    assert primary.requests[0].headers["authorization"] == "Bearer sk-primary"
    assert primary.requests[0].headers["x-request-id"] == "req_test"


async def test_logged_upstream_bodies_are_bounded_and_escaped() -> None:
    body = "first line\n2026-01-01 INFO forged log line\r\n" + "x" * 5_000
    error = await fail(make_router(Recorder(respond(500, text=body)), None))
    detail = error.failures[-1].detail
    assert "\n" not in detail and "\r" not in detail
    assert "forged log line" in detail
    assert len(detail) < 600


async def test_only_whitelisted_response_fields_are_kept() -> None:
    body = completion_body(
        "hi", system_fingerprint="fp_internal", x_provider_debug={"host": "gpu-7.internal"}
    )
    body["choices"][0]["logprobs"] = {"internal": True}
    body["usage"]["cost_usd"] = 0.12
    completion = await run(make_router(Recorder(respond(200, json_body=body)), None))

    assert completion.choices == [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ]
    assert completion.usage == {"prompt_tokens": 2, "completion_tokens": 40, "total_tokens": 42}


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}, 12),
        ({"prompt_tokens": 5, "completion_tokens": 7}, 12),
        ({"prompt_tokens": 5}, None),
        (None, None),
        # Malformed counts are dropped, never a reason to discard (and fail over) a completion.
        ({"prompt_tokens": 5, "completion_tokens": -7, "total_tokens": -2}, None),
        ({"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": "12"}, 12),
        ({"total_tokens": 1.5}, None),
        ({"total_tokens": False}, None),
        ([1, 2, 3], None),
        ("lots", None),
    ],
)
async def test_total_tokens_falls_back_sensibly(usage: object, expected: int | None) -> None:
    body = completion_body(total_tokens=None)
    if usage is not None:
        body["usage"] = usage
    completion = await run(make_router(Recorder(respond(200, json_body=body)), None))
    assert completion.route == "primary"
    assert completion.total_tokens == expected


# -- failover on 429 and timeouts ----------------------------------------------------------------


async def test_primary_429_fails_over_to_the_secondary() -> None:
    primary = Recorder(respond(429, json_body={"error": "slow down"}, headers={"retry-after": "7"}))
    secondary = Recorder(ok("from secondary", total_tokens=99))
    completion = await run(make_router(primary, secondary))

    assert completion.route == "fallback"
    assert completion.choices[0]["message"]["content"] == "from secondary"
    assert completion.total_tokens == 99
    assert [f.kind for f in completion.failed_attempts] == [FailureKind.RATE_LIMITED]
    assert completion.failed_attempts[0].status == 429

    sent = secondary.json()
    assert sent["model"] == "backup-model"
    assert secondary.requests[0].headers["authorization"] == "Bearer sk-backup"


async def test_slow_primary_fails_over_at_the_deadline() -> None:
    primary = Recorder(ok("late primary", delay=30))
    secondary = Recorder(ok("from secondary"))
    started = anyio.current_time()
    completion = await run(make_router(primary, secondary))
    elapsed = anyio.current_time() - started

    assert completion.route == "fallback"
    assert DEADLINE - 0.05 <= elapsed < DEADLINE + 0.5
    assert completion.failed_attempts[0].kind is FailureKind.TIMEOUT
    assert completion.failed_attempts[0].elapsed == pytest.approx(DEADLINE, abs=0.1)


async def test_a_late_primary_response_is_cancelled_never_used() -> None:
    # The primary would answer shortly after the deadline. Its handler must be cancelled (the
    # exchange is abandoned, not merely ignored) and its content must never surface.
    primary = Recorder(ok("PRIMARY-LATE", total_tokens=1_000, delay=DEADLINE + 0.05))
    secondary = Recorder(ok("SECONDARY", total_tokens=7))
    completion = await run(make_router(primary, secondary))

    assert completion.choices[0]["message"]["content"] == "SECONDARY"
    assert completion.total_tokens == 7
    assert (primary.cancelled, primary.completed) == (1, 0)
    await anyio.sleep(0.1)  # well past the moment the primary would have answered
    assert (primary.cancelled, primary.completed) == (1, 0)


async def test_slow_drip_body_cannot_evade_the_deadline() -> None:
    # Headers arrive at once and a byte every 50 ms never trips a per-read timeout, but the body
    # would take 5 s in total. The attempt deadline covers the body, so it still fails over.
    stream = DripStream(chunks=100, interval=0.05)
    secondary = Recorder(ok("from secondary"))
    started = anyio.current_time()
    completion = await run(make_router(Recorder(drip(stream)), secondary))
    elapsed = anyio.current_time() - started

    assert completion.route == "fallback"
    assert completion.failed_attempts[0].kind is FailureKind.TIMEOUT
    assert elapsed < DEADLINE + 0.5
    assert 0 < stream.sent < stream.chunks
    assert stream.closed  # the half-read response was closed, not leaked


async def test_timed_out_socket_is_closed_not_returned_to_the_pool() -> None:
    """Real TCP: the upstream sends headers, then drips. After the deadline it must see EOF."""
    disconnected = anyio.Event()

    async def drip_server(stream: anyio.abc.SocketStream) -> None:
        async with stream:
            request = b""
            while b"\r\n\r\n" not in request:
                request += await stream.receive()
            await stream.send(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 100000\r\n\r\n"
            )
            try:
                with anyio.fail_after(10):
                    while True:
                        await stream.send(b" ")
                        with anyio.move_on_after(0.02):
                            await stream.receive()  # request body, or EOF once the client hangs up
            except (anyio.EndOfStream, anyio.BrokenResourceError, OSError):
                disconnected.set()

    listener = await anyio.create_tcp_listener(local_host="127.0.0.1")
    port = listener.extra(anyio.abc.SocketAttribute.local_port)  # noqa: S610 (anyio, not Django)
    async with anyio.create_task_group() as tg:
        tg.start_soon(listener.serve, drip_server)
        router = ModelRouter(
            Upstream("primary", f"http://127.0.0.1:{port}/v1", "m", DEADLINE),
            Upstream("fallback", "https://backup.example/v1", "m", SECONDARY_TIMEOUT),
            transports={"fallback": Recorder(ok("from secondary")).transport},
        )
        completion = await run(router)
        assert completion.route == "fallback"
        with anyio.fail_after(3):
            await disconnected.wait()
        tg.cancel_scope.cancel()


# -- which failures are retried ------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 413, 422])
async def test_an_invalid_request_is_not_sent_to_the_secondary(status: int) -> None:
    secondary = Recorder(ok())
    error = await fail(make_router(Recorder(respond(status, json_body={"error": "nope"})), secondary))

    assert error.failures[-1].kind is FailureKind.CLIENT_ERROR
    assert error.failures[-1].status == status
    assert len(error.failures) == 1
    assert secondary.requests == []
    assert type(error) is UpstreamRejectedError and error.status == 400  # the client's problem


REFUSALS = [401, 402, 403, 404, 405, 408, 409, 410, 425, 451]


@pytest.mark.parametrize("status", REFUSALS)
async def test_a_refusing_primary_fails_over_by_default(status: int) -> None:
    # An expired provider key, a wrong model or URL, the provider's own timeout: another provider
    # may well serve the same request.
    secondary = Recorder(ok("from secondary"))
    completion = await run(make_router(Recorder(respond(status, json_body={"error": "nope"})), secondary))
    assert completion.route == "fallback"
    assert [(f.kind, f.status) for f in completion.failed_attempts] == [(FailureKind.REFUSED, status)]


@pytest.mark.parametrize("status", REFUSALS)
async def test_failover_on_refusal_can_be_disabled(status: int) -> None:
    secondary = Recorder(ok())
    policy = FailoverPolicy(on_refused=False)
    error = await fail(make_router(Recorder(respond(status, json_body={"error": "nope"})), secondary, policy))
    assert error.failures[-1].kind is FailureKind.REFUSED
    assert secondary.requests == []
    # A misconfigured provider is the operator's problem (502); a transient refusal is "retry later".
    if status in (408, 409, 425):
        assert type(error) is UpstreamUnavailableError and error.status == 503
    else:
        assert type(error) is UpstreamError and error.status == 502


RETRYABLE_BY_DEFAULT = [
    pytest.param(connection_refused(), FailureKind.CONNECTION, id="connect-error"),
    pytest.param(respond(500, text="Traceback ..."), FailureKind.SERVER_ERROR, id="500"),
    pytest.param(respond(503, text="overloaded"), FailureKind.SERVER_ERROR, id="503"),
    pytest.param(respond(200, text="<html>not json"), FailureKind.BAD_RESPONSE, id="not-json"),
    pytest.param(respond(200, json_body={"choices": []}), FailureKind.BAD_RESPONSE, id="no-choices"),
    pytest.param(respond(204), FailureKind.BAD_RESPONSE, id="204-empty"),
]


@pytest.mark.parametrize(("handler", "kind"), RETRYABLE_BY_DEFAULT)
async def test_connection_errors_and_5xx_fail_over_by_default(handler, kind: FailureKind) -> None:
    completion = await run(make_router(Recorder(handler), Recorder(ok("from secondary"))))
    assert completion.route == "fallback"
    assert completion.failed_attempts[0].kind is kind


@pytest.mark.parametrize(("handler", "kind"), RETRYABLE_BY_DEFAULT)
async def test_connection_and_server_failover_can_be_disabled(handler, kind: FailureKind) -> None:
    policy = FailoverPolicy(on_connection_error=False, on_server_error=False)
    secondary = Recorder(ok())
    error = await fail(make_router(Recorder(handler), secondary, policy))
    assert error.failures[-1].kind is kind
    assert secondary.requests == []


@pytest.mark.parametrize("kind", list(FailureKind))
def test_429_and_timeouts_always_fail_over(kind: FailureKind) -> None:
    strict = FailoverPolicy(on_connection_error=False, on_server_error=False, on_refused=False)
    assert strict.should_failover(kind) is (kind in {FailureKind.RATE_LIMITED, FailureKind.TIMEOUT})
    assert FailoverPolicy().should_failover(kind) is (kind is not FailureKind.CLIENT_ERROR)


# -- stale keep-alive connections (real sockets) -------------------------------------------------


def socket_router(url: str, secondary: Recorder, **kwargs) -> ModelRouter:
    # A generous deadline: nothing here waits on it, and the retry must fit inside it.
    return ModelRouter(
        Upstream("primary", url, "primary-model", 2.0),
        Upstream("fallback", "https://backup.example/v1", "backup-model", SECONDARY_TIMEOUT),
        transports={"fallback": secondary.transport},
        **kwargs,
    )


async def test_a_pooled_connection_closed_by_the_provider_is_retried_once_on_a_new_one(caplog) -> None:
    caplog.set_level(logging.INFO, logger="llm_router")
    # Connection 0 answers once, then is closed under the second request (the provider's idle
    # timeout firing as the connection is reused). Connection 1 answers.
    script = {(0, 0): "ok", (0, 1): "close", (1, 0): "ok"}
    secondary = Recorder(ok())
    async with (
        scripted_server(lambda c, n: script[c, n]) as server,
        socket_router(server.url, secondary) as router,
    ):
        first = await router.complete(REQUEST)
        second = await router.complete(REQUEST)

    assert (first.route, second.route) == ("primary", "primary")
    assert second.failed_attempts == ()
    assert server.requests == [(0, 0), (0, 1), (1, 0)]
    assert secondary.requests == []
    assert any("retrying once on a new connection" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("script", "calls"),
    [
        # A brand-new connection that dies: the provider may have read the request; never resend.
        pytest.param({(0, 0): "close"}, 1, id="fresh-connection"),
        # A reused connection whose response had started: the provider has processed the request.
        pytest.param({(0, 0): "ok", (0, 1): "headers-then-close"}, 2, id="response-started"),
        # The retry itself fails on its new connection: no second retry.
        pytest.param({(0, 0): "ok", (0, 1): "close", (1, 0): "close"}, 2, id="retry-fails"),
    ],
)
async def test_a_request_the_provider_may_have_processed_is_never_resent(script: dict, calls: int) -> None:
    secondary = Recorder(ok("from secondary"))
    async with (
        scripted_server(lambda c, n: script[c, n]) as server,
        socket_router(server.url, secondary) as router,
    ):
        completions = [await router.complete(REQUEST) for _ in range(calls)]

    final = completions[-1]
    assert final.route == "fallback"
    assert final.failed_attempts[0].kind is FailureKind.CONNECTION
    assert sorted(server.requests) == sorted(script)  # every scripted request exactly once
    assert len(secondary.requests) == 1


async def test_idle_pooled_connections_expire_before_typical_server_keepalive() -> None:
    assert DEFAULT_KEEPALIVE_EXPIRY_SECONDS < 5  # uvicorn and Node close idle connections after 5 s
    secondary = Recorder(ok())
    async with (
        scripted_server(lambda c, n: "ok") as server,
        socket_router(server.url, secondary, keepalive_expiry=0.1) as router,
    ):
        await router.complete(REQUEST)
        await router.complete(REQUEST)  # immediately: the pooled connection is reused
        await anyio.sleep(0.3)
        await router.complete(REQUEST)  # after the expiry: a new connection
    assert server.requests == [(0, 0), (0, 1), (1, 0)]


# -- upstream Retry-After ------------------------------------------------------------------------


async def test_the_smallest_upstream_retry_after_reaches_the_503() -> None:
    primary = Recorder(respond(429, json_body={"error": "busy"}, headers={"Retry-After": "7"}))
    secondary = Recorder(respond(429, json_body={"error": "busy"}, headers={"Retry-After": "3"}))
    error = await fail(make_router(primary, secondary))
    assert isinstance(error, UpstreamUnavailableError)
    assert error.headers == {"Retry-After": "3"}
    assert [f.retry_after for f in error.failures] == [7, 3]


async def test_an_unusable_upstream_retry_after_is_dropped() -> None:
    primary = Recorder(respond(429, text="busy", headers={"Retry-After": "soon, sk-live-SECRET"}))
    error = await fail(make_router(primary, None))
    assert isinstance(error, UpstreamUnavailableError)
    assert "Retry-After" not in error.headers
    assert "sk-live-SECRET" in error.failures[0].detail  # logged, never relayed


# -- response size cap ---------------------------------------------------------------------------


class BigStream(httpx.AsyncByteStream):
    """A body of ``chunks`` x 64 KiB that records how much of it was pulled."""

    def __init__(self, chunks: int) -> None:
        self.chunks = chunks
        self.sent = 0

    async def __aiter__(self):
        for _ in range(self.chunks):
            self.sent += 1
            yield b" " * 65_536


def sized_router(primary: Recorder, secondary: Recorder | None, limit: int) -> ModelRouter:
    router = make_router(primary, secondary)
    router.max_response_bytes = limit
    return router


async def test_an_oversized_response_is_a_bad_response_and_is_not_read_to_the_end() -> None:
    stream = BigStream(chunks=1_000)  # 64 MiB if read to the end, without a content-length

    async def big(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    completion = await run(sized_router(Recorder(big), Recorder(ok("from secondary")), limit=256 * 1024))

    assert completion.route == "fallback"
    [failure] = completion.failed_attempts
    assert failure.kind is FailureKind.BAD_RESPONSE
    assert "larger than 262144 bytes" in failure.detail
    assert stream.sent <= 5  # stopped right after the limit, not after 64 MiB


async def test_a_declared_oversized_response_is_refused_before_reading() -> None:
    body = completion_body("x" * 5_000)
    primary = Recorder(respond(200, json_body=body))
    completion = await run(sized_router(primary, Recorder(ok("from secondary")), limit=4_096))
    assert completion.route == "fallback"
    assert completion.failed_attempts[0].kind is FailureKind.BAD_RESPONSE


async def test_a_response_at_the_limit_is_accepted() -> None:
    body = json.dumps(completion_body("hello")).encode()
    primary = Recorder(respond(200, text=body.decode(), headers={"content-type": "application/json"}))
    completion = await run(sized_router(primary, None, limit=len(body)))
    assert completion.route == "primary"


async def test_an_oversized_error_body_keeps_its_status_meaning() -> None:
    primary = Recorder(respond(503, text="x" * 10_000))
    completion = await run(sized_router(primary, Recorder(ok()), limit=1_024))
    assert completion.failed_attempts[0].kind is FailureKind.SERVER_ERROR


# -- nothing works -------------------------------------------------------------------------------


async def test_both_upstreams_failing_reports_every_attempt() -> None:
    primary = Recorder(respond(429, json_body={"error": "busy"}))
    secondary = Recorder(ok(delay=30))
    started = anyio.current_time()
    error = await fail(make_router(primary, secondary))
    elapsed = anyio.current_time() - started

    assert [f.kind for f in error.failures] == [FailureKind.RATE_LIMITED, FailureKind.TIMEOUT]
    assert [f.upstream for f in error.failures] == ["primary", "fallback"]
    assert isinstance(error, UpstreamTimeoutError) and error.status == 504
    # bounded by the secondary timeout (asyncio may fire a timer up to one clock tick early)
    assert SECONDARY_TIMEOUT - 0.05 <= elapsed < SECONDARY_TIMEOUT + 0.5


async def test_without_a_secondary_the_primary_failure_is_final() -> None:
    error = await fail(make_router(Recorder(respond(429)), None))
    assert [f.kind for f in error.failures] == [FailureKind.RATE_LIMITED]
    assert isinstance(error, UpstreamUnavailableError) and error.status == 503


async def test_outer_cancellation_is_not_turned_into_a_failover() -> None:
    primary = Recorder(ok(delay=30))
    secondary = Recorder(ok())
    router = make_router(primary, secondary)
    async with router:
        with anyio.move_on_after(DEADLINE / 5) as outer:
            await router.complete(REQUEST)
    assert outer.cancelled_caught
    assert primary.cancelled == 1
    assert secondary.requests == []


# -- concurrency ---------------------------------------------------------------------------------


async def test_concurrent_requests_fail_over_independently() -> None:
    """Many in-flight requests, half with a hanging primary: each gets its own answer, in time."""
    rng = random.Random(1234)
    slow_markers: set[str] = set()

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        marker = request.headers["x-request-id"]
        if marker in slow_markers:
            await anyio.sleep(30)
        await anyio.sleep(rng.uniform(0, DEADLINE / 3))
        return httpx.Response(200, json=completion_body(f"primary:{marker}"))

    async def secondary_handler(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(rng.uniform(0, 0.05))
        return httpx.Response(200, json=completion_body(f"fallback:{request.headers['x-request-id']}"))

    primary, secondary = Recorder(primary_handler), Recorder(secondary_handler)
    markers = [f"req_{i}" for i in range(80)]
    slow_markers.update(m for m in markers if rng.random() < 0.5)
    results: dict[str, Completion] = {}

    started = anyio.current_time()
    async with make_router(primary, secondary) as router:

        async def one(marker: str) -> None:
            results[marker] = await router.complete(REQUEST, request_id=marker)

        async with anyio.create_task_group() as tg:
            for marker in markers:
                tg.start_soon(one, marker)
    elapsed = anyio.current_time() - started

    for marker, completion in results.items():
        expected_route = "fallback" if marker in slow_markers else "primary"
        assert completion.route == expected_route
        assert completion.choices[0]["message"]["content"] == f"{expected_route}:{marker}"
    assert len(results) == len(markers)
    assert primary.cancelled == len(slow_markers)
    assert len(secondary.requests) == len(slow_markers)
    assert elapsed < DEADLINE + 1.0
