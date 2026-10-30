"""Entry point: ``llm-router`` console script or ``python -m llm_router``."""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from collections.abc import Sequence

import uvicorn

from llm_router import __version__
from llm_router.app import create_app
from llm_router.logging_config import configure_logging
from llm_router.settings import Settings, load_settings

logger = logging.getLogger("llm_router")


def _on_windows() -> bool:
    return sys.platform == "win32"


def prepare_database(settings: Settings) -> None:
    """Create the limiter database (schema, WAL mode) once, before any server process starts.

    Worker processes then only open an initialised file instead of racing to create it, and a
    database that cannot be opened stops the server here with a clear message.
    """
    try:
        settings.open_limiter().close()
    except (sqlite3.Error, OSError) as exc:
        raise SystemExit(
            f"llm-router: cannot open the rate-limiter database {settings.db_path}: {exc}"
        ) from None


def main(argv: Sequence[str] | None = None) -> None:
    settings = load_settings(argv, os.environ)
    configure_logging(settings.log_level)
    logger.info(
        "starting llm-router v%s on http://%s:%d (%d worker(s), limiter db %s)",
        __version__,
        settings.host,
        settings.port,
        settings.workers,
        settings.db_path,
    )
    prepare_database(settings)
    if settings.workers > 1:
        if _on_windows():
            logger.warning(
                "--workers %d on Windows: uvicorn's multi-process mode is not reliable there and can "
                "leave concurrent connections unanswered. Use one worker on Windows, or run several "
                "workers on Linux/macOS.",
                settings.workers,
            )
        # Worker processes rebuild the app from the environment; the SQLite file is what they share.
        os.environ.update(settings.to_env())
        uvicorn.run(
            "llm_router.app:create_app_from_env",
            factory=True,
            host=settings.host,
            port=settings.port,
            workers=settings.workers,
            log_config=None,
        )
    else:
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
