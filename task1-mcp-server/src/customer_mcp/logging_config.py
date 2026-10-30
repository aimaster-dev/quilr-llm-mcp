"""Logging setup.

stdout is the JSON-RPC wire when serving over stdio, so every log record goes to stderr.
MCP clients (Claude Desktop, IDEs, inspectors) capture stderr and show it as server logs.
"""

from __future__ import annotations

import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
DEFAULT_LEVEL = "INFO"
LOG_LEVEL_ENV = "CUSTOMER_MCP_LOG_LEVEL"


def configure_logging(level: str | None = None) -> None:
    """Route all logging to stderr. Safe to call more than once.

    `level` comes from the command line, where argparse has already validated it. Otherwise the
    level is read from `CUSTOMER_MCP_LOG_LEVEL`; a value there that is not a known level is
    reported on stderr and replaced by INFO, because a typo in a host's config must not stop the
    server from starting.
    """
    rejected_env: str | None = None
    if level is None:
        configured = os.environ.get(LOG_LEVEL_ENV, "").strip()
        if configured.upper() in LOG_LEVELS:
            level = configured.upper()
        elif configured:
            rejected_env = configured
    resolved = (level or DEFAULT_LEVEL).upper()

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(resolved)
    # Python warnings are written to stderr by default; route them through logging too.
    logging.captureWarnings(True)

    if rejected_env is not None:
        logging.getLogger(__name__).warning(
            "ignoring %s=%r: not one of %s; using %s",
            LOG_LEVEL_ENV,
            rejected_env[:40],
            ", ".join(LOG_LEVELS),
            DEFAULT_LEVEL,
        )
