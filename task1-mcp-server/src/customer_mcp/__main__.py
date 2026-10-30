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
from customer_mcp.logging_config import configure_logging

logger = logging.getLogger("customer_mcp")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="customer-mcp",
        description="Customer-service MCP server speaking JSON-RPC over stdio. Logs go to stderr.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="DEBUG, INFO, WARNING or ERROR (default: $CUSTOMER_MCP_LOG_LEVEL or INFO)",
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

    import anyio

    from customer_mcp.server import create_server
    from customer_mcp.stdio import serve_stdio

    logger.info("starting %s v%s on stdio", SERVER_NAME, __version__)
    try:
        anyio.run(serve_stdio, create_server(), wire)
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")
    logger.info("stdin closed, server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
