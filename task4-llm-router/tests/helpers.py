"""Shared test helpers (importable by spawned worker processes, so keep it light)."""

from __future__ import annotations

import threading
from typing import Any

from llm_router.keys import fingerprint_api_key
from llm_router.limiter import TokenRateLimiter


class FakeClock:
    """Wall-clock stand-in: starts at a fixed epoch and only moves when told to."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


TENANT_A = fingerprint_api_key("tenant-a-key")
TENANT_B = fingerprint_api_key("tenant-b-key")


def reserve_in_child_process(
    db_path: str, limit: int, cost: int, attempts: int, barrier: Any, results: Any
) -> None:
    """Worker for the multi-process test: its own process, connection and real clock."""
    limiter = TokenRateLimiter(db_path, limit_tokens=limit)
    try:
        barrier.wait(timeout=60)
        admitted = sum(limiter.reserve(TENANT_A, cost).admitted for _ in range(attempts))
    finally:
        limiter.close()
    results.put(admitted)
