"""Entry point: `python -m customer_mcp` or the `customer-mcp` console script.

Only the standard library and `customer_mcp.claim` are imported at module level. The wire is
claimed before the SDK, pydantic or the server are imported, so even their import-time output
cannot reach stdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from customer_mcp import SERVER_NAME, __version__
from customer_mcp.claim import WireUnavailableError, claim_stdio
from customer_mcp.logging_config import LOG_LEVEL_ENV, LOG_LEVELS, configure_logging

logger = logging.getLogger("customer_mcp")

EXIT_INTERRUPTED = 130  # 128 + SIGINT, what shells report for a process stopped by Ctrl+C


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="customer-mcp",
        description="Customer-service MCP server speaking JSON-RPC over stdio. Logs go to stderr.",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default=None,
        help=f"log level, case-insensitive (default: ${LOG_LEVEL_ENV} or INFO)",
    )
    args = parser.parse_args(argv)

    # First, before logging or the server exist: from here on stdout carries only JSON-RPC.
    try:
        wire = claim_stdio()
    except WireUnavailableError as exc:
        configure_logging(args.log_level)
        logger.error("%s", exc)
        return 1
    configure_logging(args.log_level)
    for note in wire.notes:
        logger.warning("%s", note)

    try:
        import anyio

        from customer_mcp.server import create_server
        from customer_mcp.stdio import serve_stdio

        logger.info("starting %s v%s on stdio", SERVER_NAME, __version__)
        anyio.run(serve_stdio, create_server(), wire)
    except KeyboardInterrupt:
        # stdin may still be open: the read pending on it is abandoned, not waited for.
        logger.info("interrupted (Ctrl+C), server stopped")
        return EXIT_INTERRUPTED
    logger.info("stdin closed, server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
