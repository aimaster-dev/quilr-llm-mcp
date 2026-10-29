"""Logging setup shared by the gateway and the mock provider. Message content is never logged."""

from __future__ import annotations

import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> str:
    """Send logs to stderr and return the level that was applied. Safe to call more than once."""
    resolved = (level or os.environ.get("GUARDRAIL_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(resolved)
    return resolved
