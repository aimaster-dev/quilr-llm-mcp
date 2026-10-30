"""Sliding-window limiter: window semantics, eviction, Retry-After, reconciliation, atomicity."""

from __future__ import annotations

import multiprocessing
import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anyio
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from llm_router.keys import fingerprint_api_key
from llm_router.limiter import (
    DEFAULT_TOKENS_PER_WINDOW,
    DEFAULT_WINDOW_SECONDS,
    MAX_ROW_TOKENS,
    AsyncTokenRateLimiter,
    LimitDecision,
    Outcome,
    TokenRateLimiter,
)
from tests.helpers import TENANT_A, TENANT_B, FakeClock, reserve_in_child_process

# -- configuration and storage -------------------------------------------------------------------


def test_defaults_match_the_spec(make_limiter) -> None:
    assert DEFAULT_TOKENS_PER_WINDOW == 50_000
    assert DEFAULT_WINDOW_SECONDS == 60.0
    limiter = make_limiter()
    assert (limiter.limit, limiter.window) == (50_000, 60.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"limit_tokens": 0}, {"limit_tokens": 2**40 + 1}, {"window_seconds": 0}, {"busy_timeout_ms": -1}],
)
def test_rejects_invalid_configuration(make_limiter, kwargs) -> None:
    with pytest.raises(ValueError):
        make_limiter(**kwargs)


def test_database_uses_wal_and_busy_timeout(make_limiter, db_path: Path) -> None:
    limiter = make_limiter(busy_timeout_ms=1234)
    assert limiter._conn.execute("PRAGMA busy_timeout").fetchone() == (1234,)
    with sqlite3.connect(db_path) as probe:
        assert probe.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert probe.execute("PRAGMA user_version").fetchone() == (1,)
    probe.close()


def test_refuses_a_newer_schema(db_path: Path) -> None:
    raw = sqlite3.connect(db_path)
    raw.execute("PRAGMA user_version = 99")
    raw.close()
    with pytest.raises(RuntimeError, match="newer"):
        TokenRateLimiter(db_path)


def test_state_survives_restart(make_limiter, db_path: Path) -> None:
    first = make_limiter(limit_tokens=1_000)
    assert first.reserve(TENANT_A, 600).admitted
    first.close()

    second = make_limiter(limit_tokens=1_000)
    assert second.usage(TENANT_A) == 600
    assert second.reserve(TENANT_A, 500).outcome is Outcome.RATE_LIMITED


# -- admission and the sliding window ------------------------------------------------------------


def test_admits_up_to_the_limit_then_rejects(make_limiter) -> None:
    limiter = make_limiter(limit_tokens=1_000)
    decisions = [limiter.reserve(TENANT_A, 250) for _ in range(5)]

    assert [d.outcome for d in decisions] == [Outcome.ADMITTED] * 4 + [Outcome.RATE_LIMITED]
    assert [d.remaining for d in decisions] == [750, 500, 250, 0, 0]
    assert decisions[0].reservation_id != decisions[1].reservation_id
    assert decisions[-1].reservation_id is None
    assert limiter.usage(TENANT_A) == 1_000


def test_exact_fit_is_admitted(make_limiter) -> None:
    limiter = make_limiter(limit_tokens=1_000)
    assert limiter.reserve(TENANT_A, 999).admitted
    assert limiter.reserve(TENANT_A, 1).admitted
    assert not limiter.reserve(TENANT_A, 1).admitted


def test_tenants_are_isolated(make_limiter) -> None:
    limiter = make_limiter(limit_tokens=1_000)
    assert limiter.reserve(TENANT_A, 1_000).admitted
    assert not limiter.reserve(TENANT_A, 1).admitted
    assert limiter.reserve(TENANT_B, 1_000).admitted


def test_window_slides_rather_than_resetting(make_limiter, clock: FakeClock) -> None:
    limiter = make_limiter(limit_tokens=1_000, window_seconds=60)
    assert limiter.reserve(TENANT_A, 600).admitted  # t=0
    clock.advance(30)
    assert limiter.reserve(TENANT_A, 400).admitted  # t=30
    clock.advance(29.999)
    assert not limiter.reserve(TENANT_A, 1).admitted  # t=59.999: both still in the window
    clock.advance(0.001)
    # t=60: the first reservation has left the window, the second has not (no fixed-bucket reset)
    assert limiter.usage(TENANT_A) == 400
    assert limiter.reserve(TENANT_A, 600).admitted
    assert not limiter.reserve(TENANT_A, 1).admitted


def test_a_request_larger_than_the_limit_is_too_large_not_rate_limited(make_limiter) -> None:
    limiter = make_limiter(limit_tokens=1_000)
    assert limiter.reserve(TENANT_A, 300).admitted
    decision = limiter.reserve(TENANT_A, 1_001)
    assert decision.outcome is Outcome.TOO_LARGE
    assert decision.retry_after is None
    assert decision.remaining == 700
    assert limiter.usage(TENANT_A) == 300  # nothing reserved


@pytest.mark.parametrize("cost", [0, -5])
def test_rejects_non_positive_costs(make_limiter, cost: int) -> None:
    with pytest.raises(ValueError):
        make_limiter().reserve(TENANT_A, cost)


# -- Retry-After ---------------------------------------------------------------------------------


def test_retry_after_walks_the_oldest_reservations(make_limiter, clock: FakeClock) -> None:
    limiter = make_limiter(limit_tokens=60, window_seconds=60)
    start = clock()
    limiter.reserve(TENANT_A, 10)  # t=0
    clock.advance(5)
    limiter.reserve(TENANT_A, 20)  # t=5
    clock.advance(5)
    limiter.reserve(TENANT_A, 30)  # t=10
    clock.advance(10)  # t=20, window full (60/60)

    # Needs 25 tokens: the t=0 row frees 10 (not enough), the t=5 row frees 30 in total.
    decision = limiter.reserve(TENANT_A, 25)
    assert decision.outcome is Outcome.RATE_LIMITED
    assert decision.retry_after == pytest.approx(45.0)  # t=5 row expires at t=65

    clock.advance(decision.retry_after - 0.01)
    assert not limiter.reserve(TENANT_A, 25).admitted
    clock.now = start + 65
    assert limiter.reserve(TENANT_A, 25).admitted


def test_retry_after_header_rounds_up() -> None:
    def decision(retry_after: float) -> LimitDecision:
        return LimitDecision(Outcome.RATE_LIMITED, 10, 5, 10, 0, retry_after=retry_after)

    assert decision(44.2).retry_after_header == 45
    assert decision(45.0).retry_after_header == 45
    assert decision(0.0001).retry_after_header == 1
    assert LimitDecision(Outcome.ADMITTED, 10, 5, 0, 5, "r").retry_after_header is None


# -- reconciliation and release ------------------------------------------------------------------


def test_reconcile_replaces_the_estimate_with_actual_usage(make_limiter) -> None:
    limiter = make_limiter(limit_tokens=1_000)
    decision = limiter.reserve(TENANT_A, 800)
    assert limiter.reconcile(decision.reservation_id, 120) == 880
    assert limiter.usage(TENANT_A) == 120
    assert limiter.reserve(TENANT_A, 880).admitted


def test_reconcile_records_the_truth_even_above_the_limit(make_limiter, clock: FakeClock) -> None:
    limiter = make_limiter(limit_tokens=1_000, window_seconds=60)
    decision = limiter.reserve(TENANT_A, 500)
    assert limiter.reconcile(decision.reservation_id, 1_400) == 0
    assert limiter.usage(TENANT_A) == 1_400

    clock.advance(10)
    blocked = limiter.reserve(TENANT_A, 100)
    assert blocked.outcome is Outcome.RATE_LIMITED
    assert blocked.retry_after == pytest.approx(50.0)  # until the over-budget row drains
    clock.advance(50)
    assert limiter.reserve(TENANT_A, 100).admitted


def test_reconcile_of_an_evicted_reservation_is_a_no_op(make_limiter, clock: FakeClock) -> None:
    limiter = make_limiter(limit_tokens=1_000, window_seconds=60)
    decision = limiter.reserve(TENANT_A, 100)
    clock.advance(61)
    assert limiter.sweep() == 1
    assert limiter.reconcile(decision.reservation_id, 50) is None
    assert limiter.row_count() == 0


def test_release_frees_the_reservation_once(make_limiter) -> None:
    limiter = make_limiter(limit_tokens=1_000)
    decision = limiter.reserve(TENANT_A, 1_000)
    assert not limiter.reserve(TENANT_A, 1).admitted
    assert limiter.release(decision.reservation_id) is True
    assert limiter.release(decision.reservation_id) is False
    assert limiter.reserve(TENANT_A, 1_000).admitted


@pytest.mark.parametrize("tokens", [-1, MAX_ROW_TOKENS + 1, 2**63])
def test_reconcile_rejects_out_of_range_usage(make_limiter, tokens: int) -> None:
    # 2**63 would raise OverflowError inside sqlite3; anything above MAX_ROW_TOKENS could later make
    # SUM() overflow and lock the tenant out. Both are refused up front and the row is unchanged.
    limiter = make_limiter()
    decision = limiter.reserve(TENANT_A, 10)
    with pytest.raises(ValueError):
        limiter.reconcile(decision.reservation_id, tokens)
    assert limiter.usage(TENANT_A) == 10


# -- eviction ------------------------------------------------------------------------------------


def test_hot_path_evicts_only_the_requesting_tenant(make_limiter, clock: FakeClock) -> None:
    limiter = make_limiter(limit_tokens=1_000, window_seconds=60)
    for _ in range(3):
        limiter.reserve(TENANT_A, 10)
        limiter.reserve(TENANT_B, 10)
    clock.advance(60)

    limiter.reserve(TENANT_B, 10)
    # B's three expired rows were deleted in the reserve transaction; A's are left for the sweeper.
    assert limiter.row_count() == 3 + 1
    assert limiter.usage(TENANT_A) == 0  # expired rows never count, even before deletion
    assert limiter.sweep() == 3
    assert limiter.row_count() == 1


@pytest.mark.anyio
async def test_background_sweeper_evicts_idle_tenants(make_limiter, clock: FakeClock) -> None:
    limiter = AsyncTokenRateLimiter(make_limiter(limit_tokens=1_000, window_seconds=60))
    await limiter.reserve(TENANT_A, 10)
    await limiter.reserve(TENANT_B, 10)
    clock.advance(61)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(limiter.run_sweeper, 0.01)
            while limiter.sync.row_count():
                await anyio.sleep(0.01)
            tg.cancel_scope.cancel()


# -- API key fingerprints ------------------------------------------------------------------------


def test_only_key_fingerprints_reach_the_database(make_limiter, tmp_path: Path) -> None:
    raw_key = "sk-tenant-RAW-SECRET-0123456789"
    tenant = fingerprint_api_key(raw_key)
    limiter = make_limiter()
    with pytest.raises(ValueError, match="fingerprint"):
        limiter.reserve(raw_key, 10)
    assert limiter.reserve(tenant, 10).admitted

    def stored_bytes() -> bytes:
        return b"".join(p.read_bytes() for p in sorted(tmp_path.glob("limiter.db*")))

    assert tenant.encode() in stored_bytes()  # sanity: we are looking at the right files
    assert raw_key.encode() not in stored_bytes()
    limiter.close()  # checkpoint the WAL into the main file and look again
    assert raw_key.encode() not in stored_bytes()


# -- atomicity under concurrency -----------------------------------------------------------------


def test_separate_connections_in_threads_admit_exactly_the_limit(make_limiter) -> None:
    """Each thread has its own connection (like separate processes): only SQLite's lock protects us."""
    workers, attempts, cost, limit = 8, 6, 100, 1_000
    limiters = [make_limiter(limit_tokens=limit) for _ in range(workers)]
    barrier = threading.Barrier(workers)

    def worker(limiter: TokenRateLimiter) -> int:
        barrier.wait(timeout=10)
        return sum(limiter.reserve(TENANT_A, cost).admitted for _ in range(attempts))

    with ThreadPoolExecutor(workers) as pool:
        admitted = sum(pool.map(worker, limiters))

    assert admitted == limit // cost
    assert limiters[0].usage(TENANT_A) == limit


def test_separate_processes_admit_exactly_the_limit(make_limiter, db_path: Path) -> None:
    make_limiter().close()  # create the schema and switch to WAL before the race starts
    processes, attempts, cost, limit = 4, 8, 100, 1_000
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(processes)
    results = ctx.Queue()
    children = [
        ctx.Process(
            target=reserve_in_child_process, args=(str(db_path), limit, cost, attempts, barrier, results)
        )
        for _ in range(processes)
    ]
    for child in children:
        child.start()
    admitted = [results.get(timeout=60) for _ in children]
    for child in children:
        child.join(timeout=60)
        assert child.exitcode == 0

    assert sum(admitted) == limit // cost


@pytest.mark.anyio
async def test_async_facade_is_atomic_and_runs_off_the_event_loop(make_limiter, clock: FakeClock) -> None:
    loop_thread = threading.get_ident()
    seen_threads: set[int] = set()

    def recording_clock() -> float:
        seen_threads.add(threading.get_ident())
        return clock()

    limiter = AsyncTokenRateLimiter(make_limiter(limit_tokens=1_000, clock=recording_clock))
    decisions: list[LimitDecision] = []

    async def attempt() -> None:
        decisions.append(await limiter.reserve(TENANT_A, 100))

    async with anyio.create_task_group() as tg:
        for _ in range(40):
            tg.start_soon(attempt)

    assert sum(d.admitted for d in decisions) == 10
    assert seen_threads and loop_thread not in seen_threads


# -- property: decisions match the definition of a sliding window -------------------------------

LIMIT, WINDOW = 100, 60.0


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ops=st.lists(
        st.tuples(
            st.integers(0, 120).map(lambda quarters: quarters / 4),  # exact binary fractions
            st.integers(1, LIMIT),
            st.one_of(st.none(), st.integers(0, 2 * LIMIT)),
        ),
        min_size=1,
        max_size=30,
    )
)
def test_decisions_match_sliding_window_semantics(ops: list[tuple[float, int, int | None]]) -> None:
    """Checks every decision against the definition, not against a copy of the algorithm.

    * admitted  <=> live tokens + cost <= limit
    * Retry-After is exact: the request fits at now + retry_after and not a moment earlier.
    """
    clock = FakeClock()
    live: dict[str, tuple[float, int]] = {}

    def window_sum(at: float) -> int:
        return sum(tokens for ts, tokens in live.values() if ts > at - WINDOW)

    with (
        tempfile.TemporaryDirectory() as tmp,
        TokenRateLimiter(
            Path(tmp) / "p.db", limit_tokens=LIMIT, window_seconds=WINDOW, clock=clock
        ) as limiter,
    ):
        for advance, cost, actual in ops:
            clock.advance(advance)
            now = clock()
            before = window_sum(now)
            decision = limiter.reserve(TENANT_A, cost)
            assert decision.used == before

            if before + cost <= LIMIT:
                assert decision.admitted
                assert decision.remaining == LIMIT - before - cost
                live[decision.reservation_id] = (now, cost)
                if actual is not None:
                    remaining = limiter.reconcile(decision.reservation_id, actual)
                    live[decision.reservation_id] = (now, actual)
                    assert remaining == max(LIMIT - window_sum(now), 0)
            else:
                assert decision.outcome is Outcome.RATE_LIMITED
                retry_after = decision.retry_after
                assert retry_after is not None and 0 < retry_after <= WINDOW
                assert window_sum(now + retry_after) + cost <= LIMIT
                assert window_sum(now + retry_after - 1e-3) + cost > LIMIT

        assert limiter.usage(TENANT_A) == window_sum(clock())
