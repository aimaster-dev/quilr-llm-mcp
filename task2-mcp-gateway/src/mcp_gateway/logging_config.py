"""Logging setup: everything goes to stderr; audit records as bare JSON lines."""

from __future__ import annotations

import logging
import os
import sys

from mcp_gateway.audit import AUDIT_LOGGER_NAME

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

    # Audit lines are complete JSON documents (they carry their own timestamp), one per line, so
    # a log shipper can parse them without a format-specific pattern. They are always on: the
    # audit trail must not disappear because the operational log level was raised to WARNING.
    audit_handler = logging.StreamHandler(stream=sys.stderr)
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    audit = logging.getLogger(AUDIT_LOGGER_NAME)
    audit.handlers[:] = [audit_handler]
    audit.setLevel(logging.INFO)
    audit.propagate = False
