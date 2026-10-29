"""Logging setup: every record carries the request id of the request being served.

The id lives in a context variable set by the request middleware. Context variables follow the
request into child tasks and into anyio worker threads, so limiter and router log lines are
correlated with the error payload the client received.
"""

from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("llm_router_request_id", default="-")

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(request_id)s] %(message)s"


class RequestIdFilter(logging.Filter):
    """Adds ``record.request_id`` from the current context (``-`` outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


def configure_logging(level: str | None = None) -> None:
    """Log to stderr with request ids. Safe to call more than once."""
    resolved = (level or os.environ.get("LLM_ROUTER_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(resolved)
    logging.captureWarnings(True)
