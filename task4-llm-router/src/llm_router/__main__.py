"""Entry point: ``llm-router`` console script or ``python -m llm_router``."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import uvicorn

from llm_router import __version__
from llm_router.app import create_app
from llm_router.logging_config import configure_logging
from llm_router.settings import load_settings

logger = logging.getLogger("llm_router")


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
    if settings.workers > 1:
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
