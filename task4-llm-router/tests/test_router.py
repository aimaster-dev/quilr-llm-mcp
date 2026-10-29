"""Model router: deadline enforcement, failover decisions, cancellation and concurrency."""

from __future__ import annotations

import random

import anyio
import anyio.abc
import httpx
import pytest

from llm_router.router import (
    DEFAULT_PRIMARY_TIMEOUT_SECONDS,
    Completion,
    FailoverPolicy,
    FailureKind,
    ModelRouter,
    Upstream,
    UpstreamFailedError,
)
from tests.upstreams import (
    DripStream,
    Recorder,
    completion_body,
    connection_refused,
    drip,
    ok,
    respond,
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


async def fail(router: ModelRouter) -> UpstreamFailedError:
    with pytest.raises(UpstreamFailedError) as excinfo:
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
    ],
)
async def test_total_tokens_falls_back_sensibly(usage: dict | None, expected: int | None) -> None:
    body = completion_body(total_tokens=None)
    if usage is not None:
        body["usage"] = usage
    completion = await run(make_router(Recorder(respond(200, json_body=body)), None))
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
    # Headers arrive at once and a byte every 20 ms never trips a per-read timeout, but the body
    # would take 4 s in total. The attempt deadline covers the body, so it still fails over.
    stream = DripStream(chunks=200, interval=0.02)
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


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
async def test_non_retryable_4xx_is_not_sent_to_the_secondary(status: int) -> None:
    secondary = Recorder(ok())
    error = await fail(make_router(Recorder(respond(status, json_body={"error": "nope"})), secondary))

    assert error.final.kind is FailureKind.CLIENT_ERROR
    assert error.final.status == status
    assert len(error.failures) == 1
    assert secondary.requests == []


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
    assert error.final.kind is kind
    assert secondary.requests == []


@pytest.mark.parametrize("kind", list(FailureKind))
def test_429_and_timeouts_always_fail_over(kind: FailureKind) -> None:
    strict = FailoverPolicy(on_connection_error=False, on_server_error=False)
    assert strict.should_failover(kind) is (kind in {FailureKind.RATE_LIMITED, FailureKind.TIMEOUT})


# -- nothing works -------------------------------------------------------------------------------


async def test_both_upstreams_failing_reports_every_attempt() -> None:
    primary = Recorder(respond(429, json_body={"error": "busy"}))
    secondary = Recorder(ok(delay=30))
    started = anyio.current_time()
    error = await fail(make_router(primary, secondary))
    elapsed = anyio.current_time() - started

    assert [f.kind for f in error.failures] == [FailureKind.RATE_LIMITED, FailureKind.TIMEOUT]
    assert [f.upstream for f in error.failures] == ["primary", "fallback"]
    assert error.final.kind is FailureKind.TIMEOUT
    # bounded by the secondary timeout (asyncio may fire a timer up to one clock tick early)
    assert SECONDARY_TIMEOUT - 0.05 <= elapsed < SECONDARY_TIMEOUT + 0.5


async def test_without_a_secondary_the_primary_failure_is_final() -> None:
    error = await fail(make_router(Recorder(respond(429)), None))
    assert [f.kind for f in error.failures] == [FailureKind.RATE_LIMITED]


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
