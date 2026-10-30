"""Logging setup: everything goes to stderr; audit records as bare JSON lines."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from mcp_gateway.audit import AUDIT_LOGGER_NAME
from mcp_gateway.config import ConfigError

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_LEVEL_ENV = "MCP_GATEWAY_LOG_LEVEL"


def resolve_log_level(level: str | None = None) -> str:
    """`level`, else $MCP_GATEWAY_LOG_LEVEL, else INFO: upper-cased and checked against LOG_LEVELS."""
    resolved = (level or os.environ.get(LOG_LEVEL_ENV) or "INFO").upper()
    if resolved not in LOG_LEVELS:
        source = "--log-level" if level else f"${LOG_LEVEL_ENV}"
        raise ConfigError(f"{source} must be one of {', '.join(LOG_LEVELS)}")
    return resolved


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """`--log-level`, case-insensitive. Its environment fallback is checked by `setup_logging`:
    argparse applies `choices` to command-line values only, not to defaults."""
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default=None,
        help=f"operational log level (default: ${LOG_LEVEL_ENV} or INFO)",
    )


def setup_logging(parser: argparse.ArgumentParser, level: str | None) -> None:
    """`configure_logging` for a CLI: a bad $MCP_GATEWAY_LOG_LEVEL is a usage error, not a traceback."""
    try:
        configure_logging(level)
    except ConfigError as exc:
        parser.error(str(exc))


def configure_logging(level: str | None = None) -> None:
    """Route all logging to stderr. Safe to call more than once. Raises ConfigError for a bad level."""
    resolved = resolve_log_level(level)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(resolved)
    logging.captureWarnings(True)
    # httpx logs every downstream request at INFO; the audit log already covers that.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Audit lines are complete JSON documents (they carry their own timestamp), one per line, so
    # a log shipper can parse them without a format-specific pattern. They are always on: the
    # audit trail must not disappear because the operational log level was raised to WARNING.
    audit_handler = logging.StreamHandler(stream=sys.stderr)
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    audit = logging.getLogger(AUDIT_LOGGER_NAME)
    audit.handlers[:] = [audit_handler]
    audit.setLevel(logging.INFO)
    audit.propagate = False
