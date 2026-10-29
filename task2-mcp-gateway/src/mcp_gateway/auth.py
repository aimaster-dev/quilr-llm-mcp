"""Bearer token authentication: HS256 JWTs that carry the caller's subject and role.

Verification is strict and fails closed:

- `alg` is pinned to HS256, so `none`, HS512 and RS256/ES256 (algorithm confusion) are rejected
  before any signature check.
- The signature, `exp`, `nbf`, `iat`, `iss` and `aud` are verified, and `exp`, `iat`, `iss`, `aud`,
  `sub` and `role` must all be present.
- `role` must be exactly `admin` or `viewer`. `sub` must be printable ASCII, because it is
  forwarded downstream in an HTTP header.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import jwt

ALGORITHM = "HS256"
REALM = "mcp-gateway"
MAX_TOKEN_LENGTH = 4096
REQUIRED_CLAIMS = ("exp", "iat", "iss", "aud", "sub", "role")

# RFC 6750 section 2.1: "Bearer" 1*SP b68token. The scheme name is case-insensitive (RFC 9110).
_BEARER = re.compile(r"Bearer +([A-Za-z0-9\-._~+/]+=*) *", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"[\x21-\x7e]{1,256}")


class Role(StrEnum):
    ADMIN = "admin"
    VIEWER = "viewer"


_ROLE_VALUES = frozenset(role.value for role in Role)


@dataclass(frozen=True)
class Identity:
    """A verified caller."""

    subject: str
    role: Role


class AuthenticationError(Exception):
    """The request has no usable bearer token.

    `error` and `description` go to the client (RFC 6750 section 3). `reason` is the precise cause,
    for the audit log only: telling a client *why* its token failed would help it forge one.
    """

    def __init__(
        self,
        reason: str,
        *,
        error: str | None = "invalid_token",
        description: str = "The access token is invalid",
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.error = error
        self.description = description

    @property
    def http_status(self) -> int:
        return 400 if self.error == "invalid_request" else 401

    @property
    def www_authenticate(self) -> str:
        # RFC 6750 section 3.1: a request with no credentials at all gets no error code.
        challenge = f'Bearer realm="{REALM}"'
        if self.error is None:
            return challenge
        return f'{challenge}, error="{self.error}", error_description="{self.description}"'


def bearer_token(authorization: Sequence[str]) -> str:
    """Extract the token from the request's `Authorization` header value(s)."""
    if not authorization:
        raise AuthenticationError(
            "no Authorization header", error=None, description="A bearer token is required"
        )
    if len(authorization) > 1:
        raise AuthenticationError(
            "more than one Authorization header",
            error="invalid_request",
            description="Send exactly one Authorization header",
        )
    value = authorization[0]
    if not value.lower().startswith("bearer"):
        raise AuthenticationError(
            "Authorization header uses another scheme", error=None, description="A bearer token is required"
        )
    match = _BEARER.fullmatch(value)
    if match is None:
        raise AuthenticationError(
            "malformed Bearer credentials",
            error="invalid_request",
            description="Expected 'Authorization: Bearer <token>'",
        )
    return match.group(1)


class TokenVerifier:
    def __init__(self, *, secret: str, issuer: str, audience: str, leeway: float = 30.0) -> None:
        self._secret = secret
        self._issuer = issuer
        self._audience = audience
        self._leeway = leeway

    def verify(self, token: str) -> Identity:
        if len(token) > MAX_TOKEN_LENGTH:
            raise AuthenticationError(f"token longer than {MAX_TOKEN_LENGTH} characters")
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[ALGORITHM],
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway,
                options={"require": list(REQUIRED_CLAIMS)},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("token expired", description="The access token expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError(f"{type(exc).__name__}: {exc}") from exc

        subject, role = claims["sub"], claims["role"]
        if not isinstance(subject, str) or not SUBJECT_PATTERN.fullmatch(subject):
            raise AuthenticationError("sub claim is not 1-256 printable ASCII characters")
        if not isinstance(role, str) or role not in _ROLE_VALUES:
            raise AuthenticationError("role claim is not 'admin' or 'viewer'")
        return Identity(subject=subject, role=Role(role))


def mint_token(
    *,
    secret: str,
    subject: str,
    role: Role,
    issuer: str,
    audience: str,
    ttl_seconds: int = 3600,
    now: int | None = None,
) -> str:
    """Create a signed development token (see the `mcp-gateway-token` CLI)."""
    issued_at = int(time.time()) if now is None else now
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "role": role.value,
        "iat": issued_at,
        "nbf": issued_at,
        "exp": issued_at + ttl_seconds,
    }
    return jwt.encode(claims, secret, algorithm=ALGORITHM, headers={"typ": "JWT"})
