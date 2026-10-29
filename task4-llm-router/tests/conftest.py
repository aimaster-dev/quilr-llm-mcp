from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from llm_router.limiter import TokenRateLimiter
from tests.helpers import FakeClock


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "limiter.db"


@pytest.fixture
def make_limiter(db_path: Path, clock: FakeClock) -> Iterator[Callable[..., TokenRateLimiter]]:
    """Factory for limiters on the test's database file.

    Every connection is closed at teardown: Windows cannot delete a SQLite file that is still open.
    """
    created: list[TokenRateLimiter] = []

    def factory(**kwargs: Any) -> TokenRateLimiter:
        kwargs.setdefault("clock", clock)
        limiter = TokenRateLimiter(kwargs.pop("db_path", db_path), **kwargs)
        created.append(limiter)
        return limiter

    yield factory
    for limiter in created:
        limiter.close()
