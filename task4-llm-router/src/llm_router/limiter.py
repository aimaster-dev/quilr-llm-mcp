"""Token-aware sliding-window rate limiter backed by an on-disk SQLite database.

Algorithm: a sliding-window *log*. Every admitted request inserts one row ``(tenant, ts, tokens)``.
A tenant's usage at time ``now`` is the sum of its rows with ``ts > now - window``. This is exact
(no fixed buckets, no smoothing), so a burst at the end of one minute and another at the start of
the next can never add up to twice the limit.

Concurrency: the check ("does this request fit?") and the reservation (INSERT) happen in one
``BEGIN IMMEDIATE`` transaction. IMMEDIATE takes SQLite's write lock up front, so two connections,
threads or worker processes can never both read the same usage and both insert. WAL mode lets
readers proceed while a writer holds the lock, and ``busy_timeout`` makes a contending writer wait
for the lock instead of failing.

Availability: the busy timeout is short (1 s), and after any SQLite failure the async facade fails
every call immediately for a short window (1 s) instead of letting each queued call wait out its
own busy timeout behind the single limiter thread. Callers then get a quick 503, not one after
``k`` busy timeouts. Opening a brand-new database is retried while another process is creating it
(switching to WAL mode is not covered by ``busy_timeout``).

Lifecycle of a reservation:

1. ``reserve`` inserts ``prompt_estimate + max_tokens`` for the tenant (or rejects the request).
2. ``reconcile`` replaces the estimate with the provider's reported usage once a completion is
   returned. It records the truth even if that pushes the tenant over the limit; later requests
   are then rejected until enough of the window drains.
3. ``release`` deletes the reservation when no completion was produced, so failed requests do not
   consume quota.

Rows older than the window are deleted on the hot path (only the requesting tenant's rows, inside
the same transaction) and by a periodic sweeper (all tenants, for keys that went quiet).
"""

from __future__ import annotations

import logging
import math
import random
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

import anyio
import anyio.to_thread

from llm_router.keys import is_fingerprint

logger = logging.getLogger(__name__)

DEFAULT_TOKENS_PER_WINDOW = 50_000
DEFAULT_WINDOW_SECONDS = 60.0
#: How long one transaction waits for SQLite's write lock. Real transactions hold it for well under
#: a millisecond, so waiting longer only delays the 503 when the database is genuinely stuck.
DEFAULT_BUSY_TIMEOUT_MS = 1_000
#: After a failure, every limiter call fails immediately for this long (see AsyncTokenRateLimiter).
DEFAULT_FAIL_FAST_SECONDS = 1.0
#: How long opening a database keeps retrying while other processes are creating it.
INIT_RETRY_SECONDS = 10.0
SCHEMA_VERSION = 1
#: Largest token count a row may hold: far above any real request, and small enough that SUM()
#: over a tenant's rows can never overflow SQLite's 64-bit INTEGER (which would make every later
#: reservation for that tenant fail).
MAX_ROW_TOKENS = 2**40

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS reservations (
        id      TEXT PRIMARY KEY,
        tenant  TEXT NOT NULL,
        ts      REAL NOT NULL,
        tokens  INTEGER NOT NULL CHECK (tokens >= 0)
    )
    """,
    "CREATE INDEX IF NOT EXISTS reservations_tenant_ts ON reservations (tenant, ts)",
    "CREATE INDEX IF NOT EXISTS reservations_ts ON reservations (ts)",
)

Clock = Callable[[], float]
T = TypeVar("T")


class LimiterUnavailableError(RuntimeError):
    """The limiter database failed moments ago, so this call failed fast without touching it."""


def is_busy_error(exc: sqlite3.Error) -> bool:
    """SQLITE_BUSY / SQLITE_LOCKED (including extended codes): another connection holds a lock."""
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        return code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
    message = str(exc).lower()
    return "locked" in message or "busy" in message


class Outcome(StrEnum):
    ADMITTED = "admitted"
    RATE_LIMITED = "rate_limited"
    TOO_LARGE = "too_large"


@dataclass(frozen=True, slots=True)
class LimitDecision:
    outcome: Outcome
    limit: int
    cost: int
    #: Tokens in the tenant's window *before* this request.
    used: int
    #: Tokens left in the window after this decision (never negative).
    remaining: int
    reservation_id: str | None = None
    #: Seconds until enough of the window drains for this request to fit (rate-limited only).
    retry_after: float | None = None

    @property
    def admitted(self) -> bool:
        return self.outcome is Outcome.ADMITTED

    @property
    def retry_after_header(self) -> int | None:
        """``Retry-After`` value in whole seconds, rounded up so waiting that long is enough."""
        if self.retry_after is None:
            return None
        return max(1, math.ceil(self.retry_after))


class TokenRateLimiter:
    """Synchronous, thread-safe limiter. One instance owns one SQLite connection.

    Several instances (in one or many processes) may share the same database file; SQLite's
    write lock keeps check-and-reserve atomic across all of them.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        limit_tokens: int = DEFAULT_TOKENS_PER_WINDOW,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        clock: Clock = time.time,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if not 1 <= limit_tokens <= MAX_ROW_TOKENS:
            raise ValueError(f"limit_tokens must be between 1 and {MAX_ROW_TOKENS}")
        if not (math.isfinite(window_seconds) and window_seconds > 0):
            raise ValueError("window_seconds must be a positive, finite number")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must not be negative")
        self.db_path = Path(db_path)
        self.limit = int(limit_tokens)
        self.window = float(window_seconds)
        # Wall-clock time, not monotonic: timestamps are shared by every process using the file.
        self._clock = clock
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open(busy_timeout_ms)

    # -- connection management ---------------------------------------------------------------

    def _open(self, busy_timeout_ms: int) -> sqlite3.Connection:
        """Connect and initialise, retrying while other connections are initialising too.

        When several processes open a brand-new file at once, ``PRAGMA journal_mode = WAL`` can
        fail with "database is locked" straight away: changing the journal mode needs an exclusive
        lock and is not covered by ``busy_timeout``. Initialisation is idempotent, so it is simply
        retried with jittered exponential backoff.
        """
        deadline = time.monotonic() + INIT_RETRY_SECONDS
        delay = 0.005
        while True:
            try:
                return self._connect(busy_timeout_ms)
            except sqlite3.OperationalError as exc:
                if not is_busy_error(exc) or time.monotonic() + delay > deadline:
                    raise
                logger.debug("rate-limiter database busy while opening (%s); retrying", exc)
            time.sleep(delay * random.uniform(0.5, 1.5))  # noqa: S311 - jitter, not security
            delay = min(delay * 2, 0.2)

    def _connect(self, busy_timeout_ms: int) -> sqlite3.Connection:
        # isolation_level=None: the sqlite3 module never opens transactions implicitly; we issue
        # BEGIN IMMEDIATE / COMMIT ourselves. check_same_thread=False because the async wrapper
        # calls in from worker threads; self._lock guarantees one thread at a time.
        conn = sqlite3.connect(
            self.db_path,
            timeout=busy_timeout_ms / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            (mode,) = conn.execute("PRAGMA journal_mode = WAL").fetchone()
            if str(mode).lower() != "wal":
                logger.warning("SQLite refused WAL mode (journal_mode=%s); continuing", mode)
            conn.execute("PRAGMA synchronous = NORMAL")
            with self._immediate(conn):
                (version,) = conn.execute("PRAGMA user_version").fetchone()
                if version > SCHEMA_VERSION:
                    raise RuntimeError(
                        f"rate-limiter database schema v{version} is newer than supported v{SCHEMA_VERSION}"
                    )
                for statement in _SCHEMA:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except BaseException:
            conn.close()
            raise
        return conn

    @staticmethod
    @contextmanager
    def _immediate(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        """A write transaction that holds SQLite's write lock from its first statement."""
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> TokenRateLimiter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- limiter operations ------------------------------------------------------------------

    @staticmethod
    def _check_tenant(tenant: str) -> None:
        # Defence in depth: only fingerprints are ever written, so a raw key cannot end up on disk
        # through a programming mistake upstream of the limiter.
        if not is_fingerprint(tenant):
            raise ValueError("tenant must be an API key fingerprint (64 lowercase hex characters)")

    def reserve(self, tenant: str, cost: int) -> LimitDecision:
        """Atomically admit and reserve ``cost`` tokens for ``tenant``, or explain the rejection."""
        self._check_tenant(tenant)
        if cost < 1:
            raise ValueError("cost must be at least 1 token")
        if cost > self.limit:
            # Can never fit, however long the client waits: not a 429.
            used = self.usage(tenant)
            return LimitDecision(
                Outcome.TOO_LARGE, self.limit, cost, used=used, remaining=max(self.limit - used, 0)
            )

        with self._lock, self._immediate(self._conn) as conn:
            # Read the clock only once the write lock is held: time spent waiting for the lock must
            # not leave us evicting or stamping with a stale "now".
            now = self._clock()
            conn.execute("DELETE FROM reservations WHERE tenant = ? AND ts <= ?", (tenant, now - self.window))
            (used,) = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM reservations WHERE tenant = ?", (tenant,)
            ).fetchone()
            if used + cost <= self.limit:
                reservation_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO reservations (id, tenant, ts, tokens) VALUES (?, ?, ?, ?)",
                    (reservation_id, tenant, now, cost),
                )
                return LimitDecision(
                    Outcome.ADMITTED,
                    self.limit,
                    cost,
                    used=used,
                    remaining=self.limit - used - cost,
                    reservation_id=reservation_id,
                )
            rows = conn.execute(
                "SELECT ts, tokens FROM reservations WHERE tenant = ? ORDER BY ts", (tenant,)
            ).fetchall()

        return LimitDecision(
            Outcome.RATE_LIMITED,
            self.limit,
            cost,
            used=used,
            remaining=max(self.limit - used, 0),
            retry_after=self._retry_after(rows, excess=used + cost - self.limit, now=now),
        )

    def _retry_after(self, rows: Sequence[tuple[float, int]], *, excess: int, now: float) -> float:
        """Seconds until the oldest reservations free at least ``excess`` tokens.

        Rows leave the window oldest first, so walk them in timestamp order and accumulate until
        enough tokens are freed; the answer is when that row expires.
        """
        freed = 0
        for ts, tokens in rows:
            freed += tokens
            if freed >= excess:
                return max(ts + self.window - now, 0.0)
        return self.window  # unreachable while cost <= limit; a safe upper bound regardless

    def reconcile(self, reservation_id: str, actual_tokens: int) -> int | None:
        """Replace a reservation's estimate with the tokens actually used.

        Returns the tenant's remaining tokens in the current window, or ``None`` if the reservation
        no longer exists (already released, or older than the window and evicted).
        """
        if not 0 <= actual_tokens <= MAX_ROW_TOKENS:
            raise ValueError(f"actual_tokens must be between 0 and {MAX_ROW_TOKENS}")
        with self._lock, self._immediate(self._conn) as conn:
            row = conn.execute("SELECT tenant FROM reservations WHERE id = ?", (reservation_id,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE reservations SET tokens = ? WHERE id = ?", (actual_tokens, reservation_id))
            now = self._clock()
            (used,) = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM reservations WHERE tenant = ? AND ts > ?",
                (row[0], now - self.window),
            ).fetchone()
        return max(self.limit - used, 0)

    def release(self, reservation_id: str) -> bool:
        """Delete a reservation (the request failed before any completion was produced)."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM reservations WHERE id = ?", (reservation_id,))
        return cursor.rowcount > 0

    def sweep(self) -> int:
        """Delete every expired row, for all tenants. Returns the number of rows removed."""
        with self._lock, self._immediate(self._conn) as conn:
            cursor = conn.execute("DELETE FROM reservations WHERE ts <= ?", (self._clock() - self.window,))
            return cursor.rowcount

    def usage(self, tenant: str) -> int:
        """Tokens currently counted against ``tenant`` (read-only)."""
        self._check_tenant(tenant)
        with self._lock:
            (used,) = self._conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM reservations WHERE tenant = ? AND ts > ?",
                (tenant, self._clock() - self.window),
            ).fetchone()
        return int(used)

    def row_count(self) -> int:
        """Total rows stored, expired or not (for tests and diagnostics)."""
        with self._lock:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM reservations").fetchone()
        return int(count)


class AsyncTokenRateLimiter:
    """Async facade: every SQLite call runs on a worker thread, never on the event loop.

    The calls are not abandoned on cancellation (anyio's default): a transaction that started
    always finishes, and the cancellation is delivered once it has.

    Fail fast: calls queue for the single limiter thread. If the database is locked, each of them
    would wait out its own busy timeout in turn, so the k-th caller would get its error after
    k busy timeouts. Instead, once a call fails with a SQLite error, every call for the next
    ``fail_fast_seconds`` (including those already queued) raises ``LimiterUnavailableError``
    without touching the database. The first call after the window probes it again.
    """

    def __init__(
        self,
        limiter: TokenRateLimiter,
        *,
        max_threads: int = 1,
        fail_fast_seconds: float = DEFAULT_FAIL_FAST_SECONDS,
    ) -> None:
        if not (math.isfinite(fail_fast_seconds) and fail_fast_seconds >= 0):
            raise ValueError("fail_fast_seconds must be a non-negative number")
        self.sync = limiter
        # The sync limiter serialises on its own lock, so more threads would only queue up there.
        self._threads = anyio.CapacityLimiter(max_threads)
        self._fail_fast_seconds = fail_fast_seconds
        self._failed_at: float | None = None  # time.monotonic() of the last failure

    @property
    def limit(self) -> int:
        return self.sync.limit

    @property
    def window(self) -> float:
        return self.sync.window

    def _check_available(self) -> None:
        failed_at = self._failed_at
        if failed_at is not None and time.monotonic() - failed_at < self._fail_fast_seconds:
            raise LimiterUnavailableError(
                f"rate-limiter database failed {time.monotonic() - failed_at:.3f}s ago; failing fast"
            )

    def _call(self, func: Callable[..., T], *args: object) -> T:
        """Runs on the worker thread, so calls that queued behind a failure fail fast too."""
        self._check_available()
        try:
            return func(*args)
        except sqlite3.Error:
            self._failed_at = time.monotonic()
            raise

    async def _run(self, func: Callable[..., T], *args: object) -> T:
        self._check_available()  # no need to queue for the thread at all
        return await anyio.to_thread.run_sync(self._call, func, *args, limiter=self._threads)

    async def reserve(self, tenant: str, cost: int) -> LimitDecision:
        return await self._run(self.sync.reserve, tenant, cost)

    async def reconcile(self, reservation_id: str, actual_tokens: int) -> int | None:
        return await self._run(self.sync.reconcile, reservation_id, actual_tokens)

    async def release(self, reservation_id: str) -> bool:
        return await self._run(self.sync.release, reservation_id)

    async def sweep(self) -> int:
        return await self._run(self.sync.sweep)

    async def usage(self, tenant: str) -> int:
        return await self._run(self.sync.usage, tenant)

    async def run_sweeper(self, interval_seconds: float) -> None:
        """Evict expired rows every ``interval_seconds`` until cancelled."""
        while True:
            await anyio.sleep(interval_seconds)
            try:
                removed = await self.sweep()
            except (sqlite3.Error, LimiterUnavailableError):
                logger.exception("rate-limiter sweep failed; retrying next interval")
            else:
                if removed:
                    logger.debug("rate-limiter sweep evicted %d expired reservations", removed)

    def close(self) -> None:
        self.sync.close()
