"""`mcp-gateway-token`: mint development bearer tokens signed with the gateway's secret.

The secret is read from MCP_GATEWAY_JWT_SECRET only: a command-line flag would expose it in the
process list and shell history.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from mcp_gateway.auth import SUBJECT_PATTERN, Role, mint_token
from mcp_gateway.config import (
    DEFAULT_JWT_AUDIENCE,
    DEFAULT_JWT_ISSUER,
    JWT_SECRET_ENV,
    ConfigError,
    check_jwt_secret,
)

MAX_TTL_SECONDS = 7 * 24 * 3600


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mcp-gateway-token",
        description=f"Print a signed HS256 development token. The secret is read from ${JWT_SECRET_ENV}.",
    )
    parser.add_argument("--subject", required=True, help="the `sub` claim, e.g. alice")
    parser.add_argument("--role", required=True, choices=[role.value for role in Role])
    parser.add_argument("--ttl", type=int, default=3600, help="lifetime in seconds (default: %(default)s)")
    parser.add_argument(
        "--issuer", default=os.environ.get("MCP_GATEWAY_JWT_ISSUER", DEFAULT_JWT_ISSUER), help="`iss` claim"
    )
    parser.add_argument(
        "--audience",
        default=os.environ.get("MCP_GATEWAY_JWT_AUDIENCE", DEFAULT_JWT_AUDIENCE),
        help="`aud` claim",
    )
    args = parser.parse_args(argv)

    secret = os.environ.get(JWT_SECRET_ENV, "")
    try:
        check_jwt_secret(secret)
    except ConfigError as exc:
        parser.error(str(exc))
    if not 0 < args.ttl <= MAX_TTL_SECONDS:
        parser.error(f"--ttl must be between 1 and {MAX_TTL_SECONDS} seconds")
    if not SUBJECT_PATTERN.fullmatch(args.subject):
        parser.error("--subject must be 1-256 printable ASCII characters without spaces")

    token = mint_token(
        secret=secret,
        subject=args.subject,
        role=Role(args.role),
        issuer=args.issuer,
        audience=args.audience,
        ttl_seconds=args.ttl,
    )
    sys.stdout.write(token + "\n")


if __name__ == "__main__":
    main()
