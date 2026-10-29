"""Bearer token authentication: strict JWT verification, RFC 6750 challenges, no token passthrough."""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import httpx
import jwt
import pytest

from mcp_gateway.auth import AuthenticationError, Identity, Role, TokenVerifier, bearer_token
from mcp_gateway.config import DEFAULT_JWT_AUDIENCE, DEFAULT_JWT_ISSUER, JWT_SECRET_ENV
from mcp_gateway.token_cli import main as token_cli

from .helpers import TEST_SECRET, Downstream, gateway_client, token_for

PING = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
MISSING = object()

verifier = TokenVerifier(secret=TEST_SECRET, issuer=DEFAULT_JWT_ISSUER, audience=DEFAULT_JWT_AUDIENCE)


def claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    base = {
        "iss": DEFAULT_JWT_ISSUER,
        "aud": DEFAULT_JWT_AUDIENCE,
        "sub": "alice",
        "role": "viewer",
        "iat": now,
        "nbf": now,
        "exp": now + 300,
    }
    base.update(overrides)
    return {key: value for key, value in base.items() if value is not MISSING}


def signed(algorithm: str = "HS256", secret: str = TEST_SECRET, **overrides: Any) -> str:
    return jwt.encode(claims(**overrides), secret, algorithm=algorithm)


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def forged(header: dict[str, Any], signature: bytes = b"", **overrides: Any) -> str:
    """A token with an arbitrary header that no HS256 key signed."""
    segments = (json.dumps(header).encode(), json.dumps(claims(**overrides)).encode(), signature)
    return ".".join(b64url(segment) for segment in segments)


def tampered() -> str:
    """A genuine viewer token whose payload was swapped for an admin one, keeping the signature."""
    header, _, signature = token_for(Role.VIEWER).split(".")
    admin_payload = token_for(Role.ADMIN).split(".")[1]
    return f"{header}.{admin_payload}.{signature}"


# --- token verification -------------------------------------------------------------------------


@pytest.mark.parametrize("role", list(Role))
def test_valid_token_yields_identity(role: Role) -> None:
    assert verifier.verify(signed(role=role.value)) == Identity(subject="alice", role=role)


def test_clock_skew_within_leeway_is_tolerated() -> None:
    now = int(time.time())
    assert verifier.verify(signed(exp=now - 10, nbf=now + 10, iat=now + 10)).role is Role.VIEWER


def test_expired_token_is_reported_as_expired() -> None:
    with pytest.raises(AuthenticationError) as exc:
        verifier.verify(signed(exp=int(time.time()) - 120))
    assert exc.value.error == "invalid_token"
    assert exc.value.description == "The access token expired"


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(signed(nbf=int(time.time()) + 120), id="not-yet-valid"),
        pytest.param(signed(iat=int(time.time()) + 120), id="issued-in-the-future"),
        pytest.param(signed(iss="https://evil.example"), id="wrong-issuer"),
        pytest.param(signed(aud="another-service"), id="wrong-audience"),
        pytest.param(signed(secret="another-secret-0123456789abcdefghijklmnop"), id="wrong-secret"),
        pytest.param(tampered(), id="tampered-payload"),
        pytest.param(forged({"alg": "none", "typ": "JWT"}, role="admin"), id="alg-none"),
        pytest.param(forged({"alg": "None"}, role="admin"), id="alg-None"),
        pytest.param(signed("HS512", role="admin"), id="alg-HS512-same-secret"),
        pytest.param(forged({"alg": "RS256", "typ": "JWT"}, b"\x00" * 256, role="admin"), id="alg-RS256"),
        pytest.param(forged({"alg": "ES256"}, b"\x00" * 64, role="admin"), id="alg-ES256"),
        pytest.param("not-a-jwt", id="garbage"),
        pytest.param("a.b.c", id="garbage-segments"),
        pytest.param(signed(sub="x" * 5000), id="oversized"),
    ],
)
def test_invalid_tokens_are_rejected(token: str) -> None:
    with pytest.raises(AuthenticationError) as exc:
        verifier.verify(token)
    assert exc.value.error == "invalid_token"
    # The precise reason stays in the logs; the client gets a generic description.
    assert exc.value.description == "The access token is invalid"


@pytest.mark.parametrize("claim", ["exp", "iat", "iss", "aud", "sub", "role"])
def test_required_claims_must_be_present(claim: str) -> None:
    with pytest.raises(AuthenticationError):
        verifier.verify(signed(**{claim: MISSING}))


@pytest.mark.parametrize(
    "role", ["Admin", "ADMIN", " admin", "root", "", ["admin"], {"admin": True}, None, 1]
)
def test_unknown_roles_are_rejected(role: Any) -> None:
    with pytest.raises(AuthenticationError):
        verifier.verify(signed(role=role))


@pytest.mark.parametrize("subject", ["", "two words", "alice\r\nX-Gateway-Role: admin", "zoë", "x" * 257])
def test_subjects_unsafe_for_a_header_are_rejected(subject: str) -> None:
    with pytest.raises(AuthenticationError):
        verifier.verify(signed(sub=subject))


# --- Authorization header parsing ---------------------------------------------------------------


@pytest.mark.parametrize("values", [[], ["Basic dXNlcjpwYXNz"], ["Token abc"]])
def test_missing_bearer_credentials_get_a_bare_challenge(values: list[str]) -> None:
    with pytest.raises(AuthenticationError) as exc:
        bearer_token(values)
    assert exc.value.error is None
    assert exc.value.http_status == 401
    assert exc.value.www_authenticate == 'Bearer realm="mcp-gateway"'


@pytest.mark.parametrize(
    "values",
    [["Bearer"], ["Bearer "], ["Bearer a b"], ["Bearer a,b"], ["Bearertoken"], ["Bearer x", "Bearer y"]],
)
def test_malformed_bearer_credentials_are_invalid_requests(values: list[str]) -> None:
    with pytest.raises(AuthenticationError) as exc:
        bearer_token(values)
    assert exc.value.error == "invalid_request"
    assert exc.value.http_status == 400


@pytest.mark.parametrize("value", ["Bearer abc.def.ghi", "bearer abc.def.ghi", "BEARER   abc.def.ghi "])
def test_bearer_scheme_is_case_insensitive(value: str) -> None:
    assert bearer_token([value]) == "abc.def.ghi"


# --- over HTTP ----------------------------------------------------------------------------------


def assert_auth_error(response: httpx.Response, status: int) -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["www-authenticate"].startswith('Bearer realm="mcp-gateway"')
    body = response.json()
    assert body["id"] is None
    assert body["error"]["code"] == -32003
    return body


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["POST", "GET", "DELETE", "PUT"])
async def test_requests_without_a_token_never_reach_the_downstream(
    downstream: Downstream, method: str
) -> None:
    async with gateway_client(downstream, role=None) as gateway:
        response = await gateway.request(method, "/mcp", json=PING if method == "POST" else None)

    assert_auth_error(response, 401)
    assert response.headers["www-authenticate"] == 'Bearer realm="mcp-gateway"'
    assert downstream.requests == []


@pytest.mark.anyio
async def test_invalid_token_is_401_invalid_token(downstream: Downstream) -> None:
    async with gateway_client(downstream, role=None) as gateway:
        response = await gateway.post("/mcp", json=PING, headers={"Authorization": f"Bearer {tampered()}"})

    body = assert_auth_error(response, 401)
    assert 'error="invalid_token"' in response.headers["www-authenticate"]
    assert body["error"]["message"] == "The access token is invalid"
    assert downstream.requests == []


@pytest.mark.anyio
async def test_expired_token_is_401_and_says_so(downstream: Downstream) -> None:
    expired = token_for(Role.ADMIN, now=int(time.time()) - 7200)
    async with gateway_client(downstream, role=None) as gateway:
        response = await gateway.post("/mcp", json=PING, headers={"Authorization": f"Bearer {expired}"})

    assert_auth_error(response, 401)
    assert 'error_description="The access token expired"' in response.headers["www-authenticate"]


@pytest.mark.anyio
async def test_malformed_authorization_header_is_400(downstream: Downstream) -> None:
    async with gateway_client(downstream, role=None) as gateway:
        response = await gateway.post("/mcp", json=PING, headers={"Authorization": "Bearer two tokens"})

    assert_auth_error(response, 400)
    assert 'error="invalid_request"' in response.headers["www-authenticate"]
    assert downstream.requests == []


@pytest.mark.anyio
async def test_token_is_replaced_by_verified_identity_headers(downstream: Downstream) -> None:
    async with gateway_client(downstream, role=Role.VIEWER) as gateway:
        response = await gateway.post("/mcp", json=PING, headers={"Cookie": "session=abc"})

    assert response.status_code == 200
    forwarded = downstream.last.headers
    assert "authorization" not in forwarded
    assert "cookie" not in forwarded
    assert forwarded["x-gateway-subject"] == "alice"
    assert forwarded["x-gateway-role"] == "viewer"


@pytest.mark.anyio
async def test_client_supplied_identity_headers_are_discarded(downstream: Downstream) -> None:
    spoofed = {"X-Gateway-Role": "admin", "X-Gateway-Subject": "root", "X-Gateway-Impersonate": "admin"}
    async with gateway_client(downstream, role=Role.VIEWER) as gateway:
        await gateway.post("/mcp", json=PING, headers=spoofed)

    forwarded = downstream.last.headers
    assert forwarded.get_list("x-gateway-role") == ["viewer"]
    assert forwarded.get_list("x-gateway-subject") == ["alice"]
    assert "x-gateway-impersonate" not in forwarded


@pytest.mark.anyio
async def test_failures_are_logged_without_the_token(
    downstream: Downstream, caplog: pytest.LogCaptureFixture
) -> None:
    token = tampered()
    async with gateway_client(downstream, role=None) as gateway:
        with caplog.at_level(logging.INFO, logger="mcp_gateway"):
            await gateway.post("/mcp", json=PING, headers={"Authorization": f"Bearer {token}"})

    assert "InvalidSignatureError" in caplog.text
    assert token not in caplog.text
    assert token.split(".")[2] not in caplog.text


# --- mcp-gateway-token CLI ----------------------------------------------------------------------


def test_token_cli_mints_a_verifiable_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(JWT_SECRET_ENV, TEST_SECRET)
    token_cli(["--subject", "bob", "--role", "admin", "--ttl", "60"])

    token = capsys.readouterr().out.strip()
    assert verifier.verify(token) == Identity(subject="bob", role=Role.ADMIN)
    payload = jwt.decode(token, TEST_SECRET, algorithms=["HS256"], audience=DEFAULT_JWT_AUDIENCE)
    assert payload["exp"] - payload["iat"] == 60


@pytest.mark.parametrize(
    ("secret", "argv"),
    [
        (None, ["--subject", "bob", "--role", "admin"]),
        ("too-short", ["--subject", "bob", "--role", "admin"]),
        (TEST_SECRET, ["--subject", "bob", "--role", "superuser"]),
        (TEST_SECRET, ["--subject", "two words", "--role", "admin"]),
        (TEST_SECRET, ["--subject", "bob", "--role", "admin", "--ttl", "0"]),
    ],
)
def test_token_cli_rejects_bad_input(
    monkeypatch: pytest.MonkeyPatch, secret: str | None, argv: list[str]
) -> None:
    if secret is None:
        monkeypatch.delenv(JWT_SECRET_ENV, raising=False)
    else:
        monkeypatch.setenv(JWT_SECRET_ENV, secret)
    with pytest.raises(SystemExit) as exc:
        token_cli(argv)
    assert exc.value.code == 2
