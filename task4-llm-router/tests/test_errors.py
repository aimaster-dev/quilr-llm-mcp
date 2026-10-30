"""Error taxonomy and sanitization: one payload shape, no upstream details in responses."""

from __future__ import annotations

import logging
import string
from collections.abc import Iterable

import anyio
import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from llm_router.errors import (
    AuthenticationError,
    GatewayError,
    InternalError,
    InvalidRequestError,
    MethodNotAllowedError,
    NotFoundError,
    RateLimitError,
    RequestTooLargeError,
    ServiceUnavailableError,
    UpstreamError,
    UpstreamRejectedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from llm_router.logging_config import RequestIdFilter
from llm_router.middleware import EXCEPTION_HANDLERS, MIDDLEWARE, current_request_id
from llm_router.router import (
    AttemptFailure,
    FailureKind,
    ModelRouter,
    Upstream,
    gateway_error_for,
    parse_retry_after,
)
from tests.upstreams import Recorder, ok, respond

SECRET = "sk-live-UPSTREAM-SECRET-9f8e7d6c"
HOSTNAME = "gpu-node-17.inference.internal"
TRACEBACK = 'Traceback (most recent call last):\n  File "/srv/provider/app.py", line 42'
PROVIDER_HOST = "api.primary-provider.example"
LEAKS = [SECRET, HOSTNAME, "Traceback", "/srv/provider", PROVIDER_HOST, "primary-model", "backup-model"]

REQUEST = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
PAYLOAD_KEYS = {"type", "code", "message", "request_id"}


# -- catalogue -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status", "type_", "code"),
    [
        (InvalidRequestError(), 400, "invalid_request_error", "invalid_request"),
        (AuthenticationError(), 401, "authentication_error", "invalid_api_key"),
        (NotFoundError(), 404, "not_found_error", "not_found"),
        (MethodNotAllowedError(), 405, "invalid_request_error", "method_not_allowed"),
        (RequestTooLargeError(), 413, "invalid_request_error", "request_too_large"),
        (RateLimitError(3), 429, "rate_limit_error", "tenant_token_limit_exceeded"),
        (UpstreamRejectedError(), 400, "invalid_request_error", "upstream_rejected_request"),
        (UpstreamError(), 502, "upstream_error", "upstream_bad_response"),
        (UpstreamUnavailableError(), 503, "upstream_error", "upstream_unavailable"),
        (UpstreamTimeoutError(), 504, "timeout_error", "upstream_timeout"),
        (ServiceUnavailableError(), 503, "service_unavailable_error", "service_unavailable"),
        (InternalError(), 500, "internal_error", "internal_error"),
    ],
)
def test_error_catalogue(error: GatewayError, status: int, type_: str, code: str) -> None:
    payload = error.payload("req_123")
    assert error.status == status
    assert payload == {
        "error": {"type": type_, "code": code, "message": error.message, "request_id": "req_123"}
    }
    assert error.message and error.message[0].isupper()


def test_rate_limit_and_auth_errors_carry_their_headers() -> None:
    assert RateLimitError(17).headers == {"Retry-After": "17"}
    assert "17 seconds" in RateLimitError(17).message
    assert "1 second." in RateLimitError(1).message
    assert AuthenticationError().headers == {"WWW-Authenticate": "Bearer"}


def failure(kind: FailureKind, status: int | None = None, upstream: str = "primary") -> AttemptFailure:
    return AttemptFailure(upstream, kind, status=status, detail=f"{SECRET} {TRACEBACK}")


@pytest.mark.parametrize(
    ("final", "expected"),
    [
        (failure(FailureKind.TIMEOUT), UpstreamTimeoutError),
        (failure(FailureKind.RATE_LIMITED, 429), UpstreamUnavailableError),
        (failure(FailureKind.CONNECTION), UpstreamUnavailableError),
        (failure(FailureKind.SERVER_ERROR, 500), UpstreamUnavailableError),
        (failure(FailureKind.BAD_RESPONSE, 200), UpstreamError),
        (failure(FailureKind.CLIENT_ERROR, 400), UpstreamRejectedError),
        (failure(FailureKind.CLIENT_ERROR, 422), UpstreamRejectedError),
        # Refusals: a misconfigured provider is the operator's problem (502), a transient one is 503.
        (failure(FailureKind.REFUSED, 401), UpstreamError),
        (failure(FailureKind.REFUSED, 403), UpstreamError),
        (failure(FailureKind.REFUSED, 404), UpstreamError),
        (failure(FailureKind.REFUSED, 408), UpstreamUnavailableError),
        (failure(FailureKind.REFUSED, 409), UpstreamUnavailableError),
        (failure(FailureKind.REFUSED, 425), UpstreamUnavailableError),
    ],
)
def test_final_attempt_decides_the_gateway_error(final: AttemptFailure, expected: type) -> None:
    earlier = failure(FailureKind.RATE_LIMITED, 429)
    error = gateway_error_for([earlier, final])
    assert type(error) is expected
    assert error.failures == (earlier, final)
    assert SECRET not in error.message and "Traceback" not in error.message


def retry_hint(kind: FailureKind, status: int | None, retry_after: int | None) -> AttemptFailure:
    return AttemptFailure("primary", kind, status=status, detail=SECRET, retry_after=retry_after)


@pytest.mark.parametrize(
    ("failures", "header"),
    [
        # The smallest hint: when the first provider expects to be available again.
        ([(FailureKind.RATE_LIMITED, 429, 7), (FailureKind.RATE_LIMITED, 429, 3)], "3"),
        ([(FailureKind.RATE_LIMITED, 429, 7), (FailureKind.SERVER_ERROR, 500, None)], "7"),
        ([(FailureKind.SERVER_ERROR, 503, 12), (FailureKind.CONNECTION, None, None)], "12"),
        ([(FailureKind.RATE_LIMITED, 429, None), (FailureKind.RATE_LIMITED, 429, None)], None),
        # Only a 503 carries it: a final timeout (504) or bad response (502) does not.
        ([(FailureKind.RATE_LIMITED, 429, 7), (FailureKind.TIMEOUT, None, None)], None),
        ([(FailureKind.RATE_LIMITED, 429, 7), (FailureKind.BAD_RESPONSE, 200, None)], None),
    ],
)
def test_upstream_retry_after_is_relayed_on_503(failures: list, header: str | None) -> None:
    error = gateway_error_for([retry_hint(*f) for f in failures])
    assert error.headers.get("Retry-After") == header


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("7", 7),
        (" 7 ", 7),
        ("0", 1),
        ("1.2", 2),
        ("3600", 60),
        ("99999999999", 60),
        ("Thu, 01 Jan 2026 00:00:30 GMT", 30),
        ("Thu, 01 Jan 2026 00:00:00 GMT", 1),
        ("Wed, 31 Dec 2025 23:00:00 GMT", 1),
        (None, None),
        ("", None),
        ("-5", None),
        ("1e9", None),
        ("soon", None),
        ("٣", None),  # a non-ASCII digit
        ("7; drop table", None),
        ("Thu, 01 Jan 2026 00:00:30", None),  # no zone: not an HTTP-date
    ],
)
def test_upstream_retry_after_is_sanitized(value: str | None, expected: int | None) -> None:
    now = 1_767_225_600.0  # 2026-01-01T00:00:00Z
    assert parse_retry_after(value, now=now) == expected


# -- the HTTP layer ------------------------------------------------------------------------------


def make_router(primary: Recorder, secondary: Recorder | None = None, deadline: float = 0.25) -> ModelRouter:
    transports = {"primary": primary.transport}
    fallback = None
    if secondary is not None:
        transports["fallback"] = secondary.transport
        fallback = Upstream("fallback", f"https://{HOSTNAME}/v1", "backup-model", 0.5, api_key=SECRET)
    return ModelRouter(
        Upstream("primary", f"https://{PROVIDER_HOST}/v1", "primary-model", deadline, api_key=SECRET),
        fallback,
        transports=transports,
    )


def build_app(router: ModelRouter | None = None) -> Starlette:
    async def boom(request: Request) -> JSONResponse:
        raise RuntimeError(f"connect to postgres://admin:{SECRET}@{HOSTNAME}:5432 failed")

    async def wrapped(request: Request) -> JSONResponse:
        try:
            {}["missing"]
        except KeyError as exc:
            raise InternalError(log_detail=f"lookup failed on {HOSTNAME}") from exc
        return JSONResponse({})

    async def rate_limited(request: Request) -> JSONResponse:
        raise RateLimitError(17)

    async def completions(request: Request) -> JSONResponse:
        assert router is not None
        completion = await router.complete(REQUEST, request_id=current_request_id(request))
        return JSONResponse({"route": completion.route})

    return Starlette(
        routes=[
            Route("/boom", boom),
            Route("/wrapped", wrapped),
            Route("/limited", rate_limited),
            Route("/v1/chat/completions", completions, methods=["POST"]),
        ],
        middleware=MIDDLEWARE,
        exception_handlers=EXCEPTION_HANDLERS,
    )


async def call(app: Starlette, method: str, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
        return await client.request(method, path, json=REQUEST)


def assert_standard_error(response: httpx.Response, status: int, code: str) -> dict:
    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == PAYLOAD_KEYS
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    assert body["error"]["request_id"].startswith("req_")
    return body["error"]


def assert_sanitized(response: httpx.Response, leaks: Iterable[str] = LEAKS) -> None:
    exposed = (response.text + " " + " ".join(f"{k}: {v}" for k, v in response.headers.items())).lower()
    for leak in leaks:
        assert leak.lower() not in exposed, f"{leak!r} leaked into the response"


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO)
    caplog.handler.addFilter(RequestIdFilter())
    return caplog


def records_for(logs: pytest.LogCaptureFixture, request_id: str) -> list[logging.LogRecord]:
    return [r for r in logs.records if getattr(r, "request_id", None) == request_id]


@pytest.mark.anyio
async def test_unexpected_exception_becomes_a_generic_500(logs) -> None:
    response = await call(build_app(), "GET", "/boom")

    error = assert_standard_error(response, 500, "internal_error")
    assert error["type"] == "internal_error"
    assert_sanitized(response, [*LEAKS, "RuntimeError", "postgres", "admin"])
    # The full story is in the server log, under the same request id.
    [record] = records_for(logs, error["request_id"])
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert SECRET in str(record.exc_info[1])


@pytest.mark.anyio
async def test_internal_error_with_a_cause_logs_the_cause(logs) -> None:
    response = await call(build_app(), "GET", "/wrapped")
    error = assert_standard_error(response, 500, "internal_error")
    assert_sanitized(response, [*LEAKS, "KeyError", "missing"])
    [record] = records_for(logs, error["request_id"])
    assert HOSTNAME in record.getMessage()
    assert record.exc_info is not None


@pytest.mark.anyio
async def test_rate_limit_error_response(logs) -> None:
    response = await call(build_app(), "GET", "/limited")
    error = assert_standard_error(response, 429, "tenant_token_limit_exceeded")
    assert error["type"] == "rate_limit_error"
    assert response.headers["retry-after"] == "17"
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method", "path", "status", "code"),
    [("GET", "/nope", 404, "not_found"), ("GET", "/v1/chat/completions", 405, "method_not_allowed")],
)
async def test_routing_errors_use_the_standard_payload(
    method: str, path: str, status: int, code: str
) -> None:
    response = await call(build_app(), method, path)
    assert_standard_error(response, status, code)
    if status == 405:
        assert response.headers["allow"] == "POST"


@pytest.mark.anyio
async def test_failed_upstreams_are_reported_without_their_details(logs) -> None:
    primary = Recorder(respond(500, text=f"{TRACEBACK}\nKeyError: {SECRET} on {HOSTNAME}"))
    secondary = Recorder(
        respond(503, json_body={"error": {"message": f"{HOSTNAME} overloaded", "key": SECRET}})
    )
    response = await call(build_app(make_router(primary, secondary)), "POST", "/v1/chat/completions")

    error = assert_standard_error(response, 503, "upstream_unavailable")
    assert error["type"] == "upstream_error"
    assert_sanitized(response)
    # Upstream status and body snippets are in the logs, correlated by request id.
    logged = "\n".join(r.getMessage() for r in records_for(logs, error["request_id"]))
    assert SECRET in logged and HOSTNAME in logged
    assert "primary:server_error(500), fallback:server_error(503)" in logged


@pytest.mark.anyio
async def test_upstream_rejection_does_not_echo_the_upstream_body() -> None:
    body = {
        "error": {"message": f"Invalid 'messages': {SECRET} at {HOSTNAME}", "type": "invalid_request_error"}
    }
    secondary = Recorder(ok())
    app = build_app(make_router(Recorder(respond(400, json_body=body)), secondary))
    response = await call(app, "POST", "/v1/chat/completions")

    error = assert_standard_error(response, 400, "upstream_rejected_request")
    assert error["type"] == "invalid_request_error"
    assert_sanitized(response)
    assert secondary.requests == []


@pytest.mark.anyio
async def test_upstream_timeout_is_a_504() -> None:
    app = build_app(make_router(Recorder(ok(delay=30)), Recorder(ok(delay=30))))
    response = await call(app, "POST", "/v1/chat/completions")
    assert_standard_error(response, 504, "upstream_timeout")
    assert_sanitized(response)


@pytest.mark.anyio
async def test_every_response_gets_a_fresh_request_id() -> None:
    app = build_app(make_router(Recorder(ok())))
    responses = [await call(app, "POST", "/v1/chat/completions") for _ in range(3)]
    responses.append(await call(app, "GET", "/nope"))
    ids = [r.headers["x-request-id"] for r in responses]
    assert responses[0].status_code == 200
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("req_") and len(i) == 36 for i in ids)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    secret=st.text(alphabet=string.ascii_letters + string.digits, min_size=24, max_size=48),
    primary_status=st.sampled_from([429, 500, 502, 503]),
    secondary_status=st.sampled_from([400, 401, 404, 422, 429, 500, 503]),
)
def test_no_upstream_body_or_header_ever_reaches_the_client(
    secret: str, primary_status: int, secondary_status: int
) -> None:
    async def scenario() -> httpx.Response:
        primary = Recorder(
            respond(primary_status, text=f"{TRACEBACK} {secret}", headers={"x-upstream-debug": secret})
        )
        secondary = Recorder(respond(secondary_status, json_body={"error": {"message": secret}}))
        return await call(build_app(make_router(primary, secondary)), "POST", "/v1/chat/completions")

    response = anyio.run(scenario, backend="asyncio")
    assert response.status_code in {400, 502, 503}
    assert_standard_error(response, response.status_code, response.json()["error"]["code"])
    assert_sanitized(response, [secret, *LEAKS])
