"""Command-line entry points: log levels, and how uvicorn is configured to serve the gateway."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import uvicorn

from mcp_gateway import __main__ as gateway_cli
from mcp_gateway import mock_mcp_server
from mcp_gateway.audit import AUDIT_LOGGER_NAME
from mcp_gateway.auth import Role
from mcp_gateway.config import JWT_SECRET_ENV, GatewaySettings
from mcp_gateway.logging_config import LOG_LEVEL_ENV, configure_logging, resolve_log_level
from mcp_gateway.proxy import create_app

from .helpers import TEST_SECRET, serve_in_thread, token_for


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """configure_logging rewires the root and audit loggers; put pytest's capture back afterwards."""
    root, audit, httpx_logger = (logging.getLogger(name) for name in ("", AUDIT_LOGGER_NAME, "httpx"))
    saved = (
        root.handlers[:],
        root.level,
        audit.handlers[:],
        audit.level,
        audit.propagate,
        httpx_logger.level,
    )
    yield
    root.handlers[:], root.level, audit.handlers[:], audit.level, audit.propagate, httpx_logger.level = saved
    logging.captureWarnings(False)


@pytest.fixture
def uvicorn_runs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace uvicorn.run: record its keyword arguments instead of serving."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append(options))
    monkeypatch.setenv(JWT_SECRET_ENV, TEST_SECRET)
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    return calls


# --- log levels ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "env", "expected"),
    [(None, None, "INFO"), ("debug", "ERROR", "DEBUG"), (None, "warning", "WARNING")],
)
def test_log_level_comes_from_the_flag_then_the_environment(
    monkeypatch: pytest.MonkeyPatch, flag: str | None, env: str | None, expected: str
) -> None:
    if env is None:
        monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    else:
        monkeypatch.setenv(LOG_LEVEL_ENV, env)
    assert resolve_log_level(flag and flag.upper()) == expected


@pytest.mark.usefixtures("restore_logging")
def test_log_level_flag_is_case_insensitive(uvicorn_runs: list[dict[str, Any]]) -> None:
    gateway_cli.main(["--log-level", "debug"])
    assert logging.getLogger().level == logging.DEBUG
    assert len(uvicorn_runs) == 1


@pytest.mark.parametrize("main", [gateway_cli.main, mock_mcp_server.main], ids=["gateway", "mock"])
def test_an_unknown_log_level_flag_is_a_usage_error(
    uvicorn_runs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str], main: Any
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--log-level", "LOUD"])
    assert exc.value.code == 2
    assert "invalid choice: 'LOUD'" in capsys.readouterr().err
    assert uvicorn_runs == []


@pytest.mark.parametrize("main", [gateway_cli.main, mock_mcp_server.main], ids=["gateway", "mock"])
def test_an_unknown_log_level_in_the_environment_is_a_usage_error(
    uvicorn_runs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    main: Any,
) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "LOUD")
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"${LOG_LEVEL_ENV} must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL" in err
    assert "Traceback" not in err
    assert uvicorn_runs == []


def test_configure_logging_rejects_an_unknown_level_without_touching_the_loggers() -> None:
    handlers = logging.getLogger().handlers[:]
    with pytest.raises(ValueError, match="--log-level must be one of"):
        configure_logging("LOUD")
    assert logging.getLogger().handlers == handlers


# --- uvicorn access log -------------------------------------------------------------------------


@pytest.mark.usefixtures("restore_logging")
def test_the_gateway_is_served_without_an_access_log(uvicorn_runs: list[dict[str, Any]]) -> None:
    gateway_cli.main([])
    [options] = uvicorn_runs
    assert options["access_log"] is False
    assert options.items() >= gateway_cli.UVICORN_OPTIONS.items()


@pytest.mark.anyio
async def test_a_token_in_the_query_string_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Served the way `mcp-gateway` serves it, with uvicorn's loggers at INFO: the request line,
    query string included, is not written anywhere."""
    caplog.set_level(logging.DEBUG)
    token = token_for(Role.ADMIN)
    settings = GatewaySettings(jwt_secret=TEST_SECRET, downstream_url="http://127.0.0.1:9/mcp")
    with serve_in_thread(create_app(settings), log_level="info", **gateway_cli.UVICORN_OPTIONS) as url:
        async with httpx.AsyncClient(trust_env=False) as http:
            response = await http.post(
                f"{url}/mcp?access_token={token}", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}
            )

    assert response.status_code == 401  # a query-string token is not a credential either
    # Everything the server side logged (this test's own httpx client logs the URL it requested).
    server_side = [record for record in caplog.records if not record.name.startswith(("httpx", "httpcore"))]
    assert any(record.name == AUDIT_LOGGER_NAME for record in server_side)
    logged = "\n".join(record.getMessage() for record in server_side)
    assert token.split(".")[2] not in logged
    assert "access_token" not in logged
