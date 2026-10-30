"""Entry point: `python -m mcp_gateway` or the `mcp-gateway` console script."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence

import uvicorn

from mcp_gateway import GATEWAY_NAME, __version__
from mcp_gateway.config import (
    DEFAULT_DOWNSTREAM_URL,
    DEFAULT_JWT_AUDIENCE,
    DEFAULT_JWT_ISSUER,
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_STREAMS_PER_SUBJECT,
    JWT_SECRET_ENV,
    ConfigError,
    GatewaySettings,
)
from mcp_gateway.logging_config import configure_logging
from mcp_gateway.proxy import MCP_PATH, create_app

logger = logging.getLogger("mcp_gateway")


def _env(name: str, default: str) -> str:
    return os.environ.get(f"MCP_GATEWAY_{name}", default)


def build_parser() -> argparse.ArgumentParser:
    # String defaults go through `type`, so a malformed environment value is reported like a bad flag.
    parser = argparse.ArgumentParser(
        prog=GATEWAY_NAME,
        description="MCP security gateway: reverse proxy for a downstream MCP server (Streamable HTTP). "
        "Every option can also be set with the MCP_GATEWAY_<OPTION> environment variable.",
    )
    parser.add_argument(
        "--host", default=_env("HOST", "127.0.0.1"), help="bind address (default: %(default)s)"
    )
    parser.add_argument(
        "--port", type=int, default=_env("PORT", "8000"), help="bind port (default: %(default)s)"
    )
    parser.add_argument(
        "--downstream-url",
        default=_env("DOWNSTREAM_URL", DEFAULT_DOWNSTREAM_URL),
        help="downstream MCP endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=_env("CONNECT_TIMEOUT", "5"),
        help="seconds to connect to the downstream (default: %(default)s)",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=_env("READ_TIMEOUT", "60"),
        help="seconds of downstream silence tolerated on POST/DELETE (default: %(default)s)",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=_env("MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES)),
        help="largest accepted request body (default: %(default)s)",
    )
    parser.add_argument(
        "--max-streams-per-subject",
        type=int,
        default=_env("MAX_STREAMS_PER_SUBJECT", str(DEFAULT_MAX_STREAMS_PER_SUBJECT)),
        help="concurrent GET (SSE) streams one token subject may hold open (default: %(default)s)",
    )
    parser.add_argument(
        "--jwt-issuer",
        default=_env("JWT_ISSUER", DEFAULT_JWT_ISSUER),
        help="required `iss` claim (default: %(default)s)",
    )
    parser.add_argument(
        "--jwt-audience",
        default=_env("JWT_AUDIENCE", DEFAULT_JWT_AUDIENCE),
        help="required `aud` claim (default: %(default)s)",
    )
    parser.add_argument(
        "--jwt-leeway",
        type=float,
        default=_env("JWT_LEEWAY", "30"),
        help="clock skew tolerated on exp/nbf/iat, in seconds (default: %(default)s)",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING or ERROR (default: INFO)")
    parser.epilog = (
        f"The HS256 token secret is read from ${JWT_SECRET_ENV} only (at least 32 bytes), never from a flag."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        settings = GatewaySettings(
            jwt_secret=os.environ.get(JWT_SECRET_ENV, ""),
            jwt_issuer=args.jwt_issuer,
            jwt_audience=args.jwt_audience,
            jwt_leeway=args.jwt_leeway,
            downstream_url=args.downstream_url,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            max_body_bytes=args.max_body_bytes,
            max_streams_per_subject=args.max_streams_per_subject,
        )
    except ConfigError as exc:
        parser.error(str(exc))

    logger.info(
        "starting %s v%s on http://%s:%d%s -> %s",
        GATEWAY_NAME,
        __version__,
        args.host,
        args.port,
        MCP_PATH,
        settings.downstream_url,
    )
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        log_config=None,  # keep the handler installed by configure_logging
        server_header=False,
    )


if __name__ == "__main__":
    main()
