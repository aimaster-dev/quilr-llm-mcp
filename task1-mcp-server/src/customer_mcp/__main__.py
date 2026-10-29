"""Entry point: `python -m customer_mcp` or the `customer-mcp` console script."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import anyio

from customer_mcp import SERVER_NAME, __version__
from customer_mcp.logging_config import configure_logging
from customer_mcp.server import create_server
from customer_mcp.stdio import serve_stdio

logger = logging.getLogger("customer_mcp")


def main(argv: Sequence[str] | None = None) -> None:
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

    configure_logging(args.log_level)
    logger.info("starting %s v%s on stdio", SERVER_NAME, __version__)
    try:
        anyio.run(serve_stdio, create_server())
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")
    logger.info("stdin closed, server stopped")


if __name__ == "__main__":
    main()
