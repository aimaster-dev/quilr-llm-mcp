"""Logging setup: one stderr handler for the gateway and for uvicorn."""

from __future__ import annotations

import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Route all logging to stderr. Safe to call more than once."""
    resolved = (level or os.environ.get("MCP_GATEWAY_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(resolved)
    logging.captureWarnings(True)
