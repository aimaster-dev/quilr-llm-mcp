"""Logging setup.

stdout is the JSON-RPC wire when serving over stdio, so every log record goes to stderr.
MCP clients (Claude Desktop, IDEs, inspectors) capture stderr and show it as server logs.
"""

from __future__ import annotations

import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Route all logging to stderr. Safe to call more than once."""
    resolved = (level or os.environ.get("CUSTOMER_MCP_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(resolved)
    # Python warnings are written to stderr by default; route them through logging too.
    logging.captureWarnings(True)
