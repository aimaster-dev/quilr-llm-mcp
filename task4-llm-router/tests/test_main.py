"""The ``llm-router`` entry point: database preparation and multi-worker start-up."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from llm_router import __main__ as entry
from tests.test_settings import BASE_ENV


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Captures uvicorn.run calls instead of serving; logging setup is left to pytest.

    ``main`` exports its settings to ``os.environ`` for worker processes, so the environment is
    replaced by a throwaway copy for the test.
    """
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(entry.uvicorn, "run", lambda app, **kwargs: calls.append({"app": app, **kwargs}))
    monkeypatch.setattr(entry, "configure_logging", lambda level: None)
    monkeypatch.setattr(entry.os, "environ", {**os.environ, **BASE_ENV})
    return calls


def schema_of(db: Path) -> tuple[str, int]:
    probe = sqlite3.connect(db)
    try:
        (mode,) = probe.execute("PRAGMA journal_mode").fetchone()
        (version,) = probe.execute("PRAGMA user_version").fetchone()
    finally:
        probe.close()
    return mode, version


@pytest.mark.parametrize("workers", [1, 3])
def test_the_database_is_initialised_before_any_server_process_starts(
    tmp_path: Path, started: list, workers: int
) -> None:
    db = tmp_path / "nested" / "router.db"
    entry.main(["--db-path", str(db), "--workers", str(workers)])
    [call] = started
    assert schema_of(db) == ("wal", 1)  # created by the parent: workers only open it
    assert call.get("workers", 1) == workers


def test_an_unusable_database_path_is_a_clean_startup_error(tmp_path: Path, started: list) -> None:
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    with pytest.raises(SystemExit, match="cannot open the rate-limiter database"):
        entry.main(["--db-path", str(not_a_dir / "router.db")])
    assert started == []


@pytest.mark.parametrize("windows", [True, False])
def test_multiple_workers_on_windows_are_warned_about(
    tmp_path: Path, started: list, monkeypatch, caplog, windows: bool
) -> None:
    monkeypatch.setattr(entry, "_on_windows", lambda: windows)
    caplog.set_level(logging.WARNING, logger="llm_router")
    entry.main(["--db-path", str(tmp_path / "r.db"), "--workers", "2"])
    warnings = [r for r in caplog.records if "Windows" in r.getMessage()]
    assert bool(warnings) is windows
    assert len(started) == 1  # a warning, not a refusal
