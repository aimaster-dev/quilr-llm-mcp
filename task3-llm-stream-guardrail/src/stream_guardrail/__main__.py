"""Entry point: `stream-guardrail` (or `python -m stream_guardrail`) runs the gateway on uvicorn."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import uvicorn

from stream_guardrail import __version__
from stream_guardrail.config import Settings
from stream_guardrail.gateway import create_app
from stream_guardrail.logging_config import configure_logging

logger = logging.getLogger("stream_guardrail")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="stream-guardrail",
        description="OpenAI-compatible gateway that redacts PII from streamed LLM responses. "
        "The provider API key is read from $GUARDRAIL_UPSTREAM_API_KEY only.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="bind port (default: 8080)")
    parser.add_argument(
        "--upstream-base-url",
        help="provider base URL ending in /v1 (default: $GUARDRAIL_UPSTREAM_BASE_URL or "
        "http://127.0.0.1:9000/v1)",
    )
    parser.add_argument("--connect-timeout", type=float, help="seconds (default: 5)")
    parser.add_argument("--read-timeout", type=float, help="max seconds between upstream reads (default: 60)")
    parser.add_argument(
        "--log-level", help="DEBUG, INFO, WARNING or ERROR (default: $GUARDRAIL_LOG_LEVEL or INFO)"
    )
    args = parser.parse_args(argv)

    level = configure_logging(args.log_level)
    settings = Settings.from_env(
        upstream_base_url=args.upstream_base_url,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    logger.info(
        "stream-guardrail v%s -> %s (api key %s)",
        __version__,
        settings.upstream_base_url,
        "set" if settings.upstream_api_key else "not set",
    )
    uvicorn.run(
        create_app(settings), host=args.host, port=args.port, log_level=level.lower(), log_config=None
    )


if __name__ == "__main__":
    main()
