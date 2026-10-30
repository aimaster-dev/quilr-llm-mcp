"""The gateway HTTP API end to end (in-process), against scripted upstreams."""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from pathlib import Path

import anyio
import httpx
import pytest

from llm_router.keys import fingerprint_api_key
from tests.gateway import COST, KEY_A, KEY_B, PROMPT, chat_body, make_settings, running_gateway
from tests.helpers import FakeClock
from tests.upstreams import Recorder, completion_body, ok, respond

pytestmark = pytest.mark.anyio

DEADLINE = 0.25  # make_settings() primary timeout


def error_of(response: httpx.Response) -> dict:
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "code", "message", "request_id"}
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    return body["error"]


# -- authentication ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Bearer", "Bearer ", f"Basic {KEY_A}", "Bearer not-a-tenant-key", f"Bearer {KEY_A}x"],
)
async def test_requests_without_a_valid_tenant_key_are_rejected(tmp_path: Path, authorization) -> None:
    primary = Recorder(ok())
    async with running_gateway(make_settings(tmp_path), primary) as gw:
        headers = {} if authorization is None else {"Authorization": authorization}
        response = await gw.chat(key=None, headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert error_of(response)["code"] == "invalid_api_key"
    assert KEY_A not in response.text
    assert primary.requests == []


async def test_scheme_is_case_insensitive(tmp_path: Path) -> None:
    async with running_gateway(make_settings(tmp_path), Recorder(ok())) as gw:
        response = await gw.chat(key=None, headers={"Authorization": f"bearer {KEY_A}"})
    assert response.status_code == 200


# -- a successful completion ---------------------------------------------------------------------


async def test_completion_response_shape_and_headers(tmp_path: Path) -> None:
    primary = Recorder(ok("Hi there", total_tokens=37))
    async with running_gateway(make_settings(tmp_path), primary) as gw:
        response = await gw.chat()
        usage = await gw.usage()

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}
    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-test"  # the client's model name, not the provider's
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hi there"}
    assert body["usage"]["total_tokens"] == 37
    assert response.headers["x-gateway-route"] == "primary"
    assert response.headers["x-request-id"].startswith("req_")
    # Reconciled: the window holds the provider's actual usage, not the 100-token estimate.
    assert usage == 37
    assert response.headers["x-ratelimit-limit"] == "10000"
    assert response.headers["x-ratelimit-remaining"] == str(10_000 - 37)


async def test_upstream_request_is_rebuilt_not_forwarded(tmp_path: Path) -> None:
    primary = Recorder(ok())
    async with running_gateway(make_settings(tmp_path, default_max_tokens=200), primary) as gw:
        response = await gw.chat(body=chat_body(max_tokens=None, temperature=0.5))

    sent = primary.json()
    assert sent == {
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.5,
        "max_tokens": 200,  # the default is made explicit so the provider cannot exceed the reservation
        "model": "primary-model",
        "stream": False,
    }
    upstream_headers = primary.requests[0].headers
    assert upstream_headers["authorization"] == "Bearer sk-provider-primary"
    assert upstream_headers["x-request-id"] == response.headers["x-request-id"]
    assert KEY_A not in str(upstream_headers)


async def test_missing_usage_is_charged_at_the_estimate(tmp_path: Path) -> None:
    async with running_gateway(make_settings(tmp_path), Recorder(ok(total_tokens=None))) as gw:
        response = await gw.chat()
        assert response.status_code == 200
        assert "usage" not in response.json()
        assert await gw.usage() == COST


# -- reconciling with the provider's usage -------------------------------------------------------

PROMPT_ESTIMATE = 17  # see tests.gateway.PROMPT
MAX_PLAUSIBLE = (COST - PROMPT_ESTIMATE) + 4 * PROMPT_ESTIMATE  # max_tokens + 4 x prompt estimate
NO_USAGE = object()


def completion_with_usage(usage: object) -> dict:
    body = completion_body("hi", total_tokens=None)
    if usage is not NO_USAGE:
        body["usage"] = usage
    return body


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param({"total_tokens": 2**63}, id="2**63"),
        pytest.param({"total_tokens": 10**15}, id="10**15"),
        pytest.param({"total_tokens": MAX_PLAUSIBLE + 1}, id="just-above-possible"),
        pytest.param({"prompt_tokens": 9, "completion_tokens": 40_000}, id="ignored-max_tokens"),
        pytest.param({"total_tokens": 0}, id="zero"),
        pytest.param({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, id="all-zero"),
        pytest.param({"total_tokens": -5}, id="negative"),
        pytest.param({"prompt_tokens": 9, "completion_tokens": -1}, id="negative-part"),
        pytest.param({"total_tokens": "100"}, id="string"),
        pytest.param({"total_tokens": True}, id="bool"),
        pytest.param({"prompt_tokens": 9}, id="prompt-only"),
        pytest.param("n/a", id="not-an-object"),
        pytest.param(None, id="null"),
        pytest.param(NO_USAGE, id="missing"),
    ],
)
async def test_implausible_or_missing_usage_keeps_the_reservation_and_still_serves(
    tmp_path: Path, usage: object
) -> None:
    """A broken provider can neither crash the request, lock the tenant out nor make it free."""
    primary, secondary = Recorder(respond(200, json_body=completion_with_usage(usage))), Recorder(ok())
    async with running_gateway(make_settings(tmp_path), primary, secondary) as gw:
        response = await gw.chat()
        charged = await gw.usage()

    assert response.status_code == 200
    assert response.headers["x-gateway-route"] == "primary"  # a valid completion is not failed over
    assert response.json()["choices"][0]["message"]["content"] == "hi"
    assert "usage" not in response.json()  # numbers the gateway refused to charge are not relayed
    assert charged == COST
    assert response.headers["x-ratelimit-remaining"] == str(10_000 - COST)
    assert secondary.requests == []


@pytest.mark.parametrize("total", [1, PROMPT_ESTIMATE, COST - 1, COST + 1, MAX_PLAUSIBLE])
async def test_plausible_usage_is_recorded_exactly(tmp_path: Path, total: int) -> None:
    # Above the reservation is possible (the prompt estimate is not an upper bound) and recorded too.
    async with running_gateway(make_settings(tmp_path), Recorder(ok(total_tokens=total))) as gw:
        response = await gw.chat()
        charged = await gw.usage()
    assert response.status_code == 200
    assert response.json()["usage"]["total_tokens"] == total
    assert charged == total
    assert response.headers["x-ratelimit-remaining"] == str(10_000 - total)


async def test_a_failing_reconcile_still_returns_the_completion(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="llm_router")
    async with running_gateway(make_settings(tmp_path), Recorder(ok("served", total_tokens=42))) as gw:

        async def broken(*args: object) -> None:
            raise OverflowError("Python int too large to convert to SQLite INTEGER")

        monkeypatch.setattr(gw.gateway.limiter, "reconcile", broken)
        response = await gw.chat()
        charged = await gw.usage()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "served"
    assert charged == COST  # fail safe: the reservation keeps its estimate
    [record] = [r for r in caplog.records if "could not reconcile" in r.getMessage()]
    assert record.levelno == logging.ERROR and record.exc_info is not None


# -- failover through the API --------------------------------------------------------------------


async def test_primary_429_is_served_by_the_fallback(tmp_path: Path) -> None:
    primary = Recorder(respond(429, json_body={"error": "busy"}))
    secondary = Recorder(ok("from fallback", total_tokens=21))
    async with running_gateway(make_settings(tmp_path), primary, secondary) as gw:
        response = await gw.chat()
        usage = await gw.usage()

    assert response.status_code == 200
    assert response.headers["x-gateway-route"] == "fallback"
    assert response.json()["choices"][0]["message"]["content"] == "from fallback"
    assert secondary.json()["model"] == "backup-model"
    assert secondary.requests[0].headers["authorization"] == "Bearer sk-provider-backup"
    assert usage == 21


async def test_timeout_race_fails_over_at_the_deadline_and_counts_only_the_fallback(tmp_path: Path) -> None:
    # The primary would answer with a huge usage figure if it were allowed to finish.
    primary = Recorder(ok("PRIMARY", total_tokens=5_000, delay=DEADLINE + 0.1))
    secondary = Recorder(ok("FALLBACK", total_tokens=33))
    async with running_gateway(make_settings(tmp_path), primary, secondary) as gw:
        started = anyio.current_time()
        response = await gw.chat()
        elapsed = anyio.current_time() - started
        await anyio.sleep(0.2)  # past the moment the primary would have answered
        usage = await gw.usage()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "FALLBACK"
    assert DEADLINE - 0.05 <= elapsed < DEADLINE + 0.5  # bounded: deadline + fast fallback
    assert (primary.cancelled, primary.completed) == (1, 0)
    assert usage == 33  # never double-counted


@pytest.mark.parametrize(
    ("primary_handler", "secondary_handler", "status", "code"),
    [
        (respond(429), respond(429), 503, "upstream_unavailable"),
        (respond(500, text="boom"), respond(503), 503, "upstream_unavailable"),
        (respond(429), ok(delay=30), 504, "upstream_timeout"),
        (ok(delay=30), ok(delay=30), 504, "upstream_timeout"),
        (respond(400, json_body={"error": "bad"}), ok(), 400, "upstream_rejected_request"),
    ],
    ids=["429-then-429", "500-then-503", "429-then-timeout", "timeout-then-timeout", "400-not-retried"],
)
async def test_failed_requests_release_their_reservation(
    tmp_path: Path, primary_handler, secondary_handler, status: int, code: str
) -> None:
    settings = make_settings(tmp_path, secondary_timeout=0.25)
    async with running_gateway(settings, Recorder(primary_handler), Recorder(secondary_handler)) as gw:
        response = await gw.chat()
        usage = await gw.usage()

    assert response.status_code == status
    assert error_of(response)["code"] == code
    assert usage == 0
    assert response.headers["x-ratelimit-remaining"] == "10000"


async def test_client_disconnect_cancels_routing_and_releases_the_reservation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The server reports http.disconnect but does not cancel the handler (as uvicorn does)."""
    caplog.set_level(logging.INFO, logger="llm_router")
    settings = make_settings(tmp_path, primary_timeout=5.0)
    primary, secondary = Recorder(ok(delay=30)), Recorder(ok())
    sent: list = []
    async with running_gateway(settings, primary, secondary) as gw:
        hang_up = anyio.Event()

        async def client() -> None:
            sent.extend(await gw.chat_and_hang_up(hang_up))

        async with anyio.create_task_group() as tg:
            tg.start_soon(client)
            with anyio.fail_after(2):
                while not primary.requests:
                    await anyio.sleep(0.01)
            hung_up_at = anyio.current_time()
            hang_up.set()
        handler_returned_after = anyio.current_time() - hung_up_at
        usage = await gw.usage()

    assert handler_returned_after < 1.0  # not the 5 s primary deadline
    assert (primary.cancelled, primary.completed) == (1, 0)
    assert secondary.requests == []  # a client that has gone is not failed over
    assert usage == 0
    assert sent == []  # nobody to answer: nothing is sent
    [record] = [r for r in caplog.records if "client disconnected" in r.getMessage()]
    assert record.levelno == logging.INFO


async def test_outer_cancellation_releases_the_reservation(tmp_path: Path) -> None:
    """Cancellation from outside the handler (server shutdown) also releases the reservation."""
    settings = make_settings(tmp_path, primary_timeout=5.0)
    primary = Recorder(ok(delay=30))
    async with running_gateway(settings, primary) as gw:
        with anyio.move_on_after(0.2):
            await gw.chat()  # ASGITransport runs the app in this task, so this cancels the handler
        assert primary.cancelled == 1
        assert await gw.usage() == 0


# -- rate limiting -------------------------------------------------------------------------------


async def test_rate_limit_headers_and_exact_retry_after(tmp_path: Path) -> None:
    clock = FakeClock()
    settings = make_settings(tmp_path, tokens_per_window=250, window_seconds=60)
    async with running_gateway(settings, Recorder(ok(total_tokens=COST)), clock=clock) as gw:
        first = await gw.chat()  # t=0, 100 tokens
        clock.advance(10)
        second = await gw.chat()  # t=10, 200 tokens
        clock.advance(5)
        limited = await gw.chat()  # t=15: needs 50 more, freed when the t=0 request expires at t=60
        clock.advance(44.5)
        still_limited = await gw.chat()
        clock.advance(0.5)
        after_wait = await gw.chat()

    assert [r.status_code for r in (first, second)] == [200, 200]
    assert [r.headers["x-ratelimit-remaining"] for r in (first, second)] == ["150", "50"]
    assert limited.status_code == 429
    error = error_of(limited)
    assert (error["type"], error["code"]) == ("rate_limit_error", "tenant_token_limit_exceeded")
    assert limited.headers["retry-after"] == "45"
    assert limited.headers["x-ratelimit-limit"] == "250"
    assert limited.headers["x-ratelimit-remaining"] == "50"
    assert still_limited.status_code == 429
    assert still_limited.headers["retry-after"] == "1"
    assert after_wait.status_code == 200


async def test_tenants_have_separate_budgets(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, tokens_per_window=150)
    async with running_gateway(settings, Recorder(ok(total_tokens=COST))) as gw:
        statuses = [(await gw.chat(KEY_A)).status_code for _ in range(2)]
        statuses.append((await gw.chat(KEY_B)).status_code)
    assert statuses == [200, 429, 200]


async def test_a_request_that_can_never_fit_is_413_not_429(tmp_path: Path) -> None:
    primary = Recorder(ok())
    settings = make_settings(tmp_path, tokens_per_window=250)
    async with running_gateway(settings, primary) as gw:
        response = await gw.chat(body=chat_body(max_tokens=300))
        usage = await gw.usage()

    assert response.status_code == 413
    error = error_of(response)
    assert error["code"] == "request_too_large"
    assert "317" in error["message"] and "250" in error["message"]
    assert "retry-after" not in response.headers
    assert primary.requests == []
    assert usage == 0


async def test_limiter_database_failure_fails_closed(tmp_path: Path, monkeypatch) -> None:
    primary = Recorder(ok())
    async with running_gateway(make_settings(tmp_path), primary) as gw:

        async def broken(*args: object) -> None:
            raise sqlite3.OperationalError("database is locked: /var/lib/llm_router/data.db")

        monkeypatch.setattr(gw.gateway.limiter, "reserve", broken)
        response = await gw.chat()

    assert response.status_code == 503
    assert error_of(response)["code"] == "rate_limiter_unavailable"
    assert "locked" not in response.text and "/var/lib" not in response.text
    assert primary.requests == []


async def test_raw_tenant_keys_never_reach_the_database(tmp_path: Path) -> None:
    async with running_gateway(make_settings(tmp_path), Recorder(ok())) as gw:
        assert (await gw.chat(KEY_A)).status_code == 200
        assert (await gw.chat(KEY_B)).status_code == 200
        stored = b"".join(p.read_bytes() for p in tmp_path.glob("gateway.db*"))
    assert fingerprint_api_key(KEY_A).encode() in stored
    assert KEY_A.encode() not in stored and KEY_B.encode() not in stored


# -- validation ----------------------------------------------------------------------------------

SECRET_INPUT = "SECRET-INPUT-VALUE-4242"


@pytest.mark.parametrize(
    ("body", "code", "mentions"),
    [
        (b"{not json", "invalid_json", None),
        (b"[]", "invalid_request", "JSON object"),
        ({"model": "m"}, "invalid_request", "messages"),
        ({"messages": []}, "invalid_request", "messages"),
        ({"messages": SECRET_INPUT}, "invalid_request", "messages"),
        ({**chat_body(), "frequency_penalty": 1}, "invalid_request", "frequency_penalty"),
        ({**chat_body(), "max_tokens": 0}, "invalid_request", "max_tokens"),
        ({**chat_body(), "max_tokens": "100"}, "invalid_request", "max_tokens"),
        ({**chat_body(), "temperature": 5}, "invalid_request", "temperature"),
        ({"messages": [{"role": "tool", "content": SECRET_INPUT}]}, "invalid_request", "role"),
        ({"messages": [{"role": "user", "content": 42}]}, "invalid_request", "content"),
        ({**chat_body(), "stream": True}, "streaming_not_supported", "stream"),
    ],
)
async def test_invalid_requests_are_rejected_without_echoing_input(
    tmp_path: Path, body, code: str, mentions: str | None
) -> None:
    primary = Recorder(ok())
    async with running_gateway(make_settings(tmp_path), primary) as gw:
        if isinstance(body, bytes):
            response = await gw.client.post(
                "/v1/chat/completions", content=body, headers={"Authorization": f"Bearer {KEY_A}"}
            )
        else:
            response = await gw.chat(body=body)
        usage = await gw.usage()

    assert response.status_code == 400
    error = error_of(response)
    assert (error["type"], error["code"]) == ("invalid_request_error", code)
    if mentions:
        assert mentions in error["message"]
    assert SECRET_INPUT not in response.text
    assert primary.requests == []
    assert usage == 0


async def test_oversized_bodies_are_rejected(tmp_path: Path) -> None:
    primary = Recorder(ok())
    async with running_gateway(make_settings(tmp_path, max_body_bytes=2048), primary) as gw:
        body = {"messages": [{"role": "user", "content": "y" * 5_000}]}
        response = await gw.chat(body=body)
    assert response.status_code == 413
    assert error_of(response)["code"] == "payload_too_large"
    assert primary.requests == []


async def test_other_routes_use_the_standard_error_shape(tmp_path: Path) -> None:
    async with running_gateway(make_settings(tmp_path), Recorder(ok())) as gw:
        health = await gw.client.get("/healthz")
        missing = await gw.client.get("/v1/models")
        wrong_method = await gw.client.get("/v1/chat/completions")
    assert health.json() == {"status": "ok"}
    assert error_of(missing)["code"] == "not_found"
    assert error_of(wrong_method)["code"] == "method_not_allowed"


# -- concurrency ---------------------------------------------------------------------------------


async def test_parallel_requests_admit_exactly_floor_limit_over_cost(tmp_path: Path) -> None:
    limit, parallel = 1_050, 40  # floor(1050 / 100) = 10
    # The provider reports usage equal to the reservation, so reconciliation cannot free budget and
    # the outcome depends only on the atomicity of check-and-reserve.
    primary = Recorder(ok(total_tokens=COST, delay=0.05))
    async with running_gateway(make_settings(tmp_path, tokens_per_window=limit), primary) as gw:
        responses: list[httpx.Response] = []

        async def one() -> None:
            responses.append(await gw.chat())

        async with anyio.create_task_group() as tg:
            for _ in range(parallel):
                tg.start_soon(one)
        usage = await gw.usage()

    assert Counter(r.status_code for r in responses) == {200: limit // COST, 429: parallel - limit // COST}
    assert len(primary.requests) == limit // COST
    assert usage == (limit // COST) * COST


async def test_two_gateway_instances_share_one_budget(tmp_path: Path) -> None:
    """Two app instances on one database file behave like two worker processes."""
    limit, parallel = 1_000, 30
    settings = make_settings(tmp_path, tokens_per_window=limit)
    upstream = Recorder(ok(total_tokens=COST, delay=0.05))
    responses: list[httpx.Response] = []
    async with running_gateway(settings, upstream) as one, running_gateway(settings, upstream) as two:

        async def call(gw) -> None:
            responses.append(await gw.chat())

        async with anyio.create_task_group() as tg:
            for i in range(parallel):
                tg.start_soon(call, one if i % 2 else two)
        usage = await one.usage()

    assert Counter(r.status_code for r in responses)[200] == limit // COST
    assert usage == limit


async def test_concurrent_failover_under_load_keeps_answers_and_accounting_straight(tmp_path: Path) -> None:
    """Many requests at once, about half of them hitting a hanging primary: every client gets its
    own answer, from the right route, within the deadline, and is charged exactly once."""

    def hangs(request_id: str) -> bool:
        return int(request_id[-1], 16) % 2 == 1  # request ids are random: a random ~50% mix

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        request_id = request.headers["x-request-id"]
        if hangs(request_id):
            await anyio.sleep(30)
        return httpx.Response(200, json=completion_body(f"primary:{request_id}", total_tokens=10))

    async def secondary_handler(request: httpx.Request) -> httpx.Response:
        request_id = request.headers["x-request-id"]
        return httpx.Response(200, json=completion_body(f"fallback:{request_id}", total_tokens=20))

    responses: list[httpx.Response] = []
    upstreams = Recorder(primary_handler), Recorder(secondary_handler)
    async with running_gateway(make_settings(tmp_path), *upstreams) as gw:

        async def one() -> None:
            responses.append(await gw.chat())

        started = anyio.current_time()
        async with anyio.create_task_group() as tg:
            for _ in range(40):
                tg.start_soon(one)
        elapsed = anyio.current_time() - started
        usage = await gw.usage()

    routes: Counter[str] = Counter()
    for response in responses:
        request_id = response.headers["x-request-id"]
        route = "fallback" if hangs(request_id) else "primary"
        routes[route] += 1
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == route
        assert response.json()["choices"][0]["message"]["content"] == f"{route}:{request_id}"
    assert len(responses) == 40
    assert routes["primary"] and routes["fallback"]
    assert usage == routes["primary"] * 10 + routes["fallback"] * 20
    assert elapsed < DEADLINE + 1.0
