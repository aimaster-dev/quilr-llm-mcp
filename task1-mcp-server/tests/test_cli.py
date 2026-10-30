"""Command-line handling: a bad log level is reported, never a traceback."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .wire import SERVER_CMD, StdioWireClient, assert_pure_jsonrpc, server_env


def run_cli(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*SERVER_CMD, *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=server_env(env),
        cwd=Path(__file__).resolve().parent,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("level", ["BOGUS", "verbose", "10", ""])
def test_invalid_log_level_option_is_a_usage_error(level: str) -> None:
    result = run_cli(["--log-level", level])
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("level", ["debug", "Warning", "ERROR"])
def test_log_level_option_is_case_insensitive(level: str) -> None:
    result = run_cli(["--log-level", level])
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_log_level_variable_warns_and_falls_back_to_info() -> None:
    # A typo in a host's env config must not stop the server from starting.
    with StdioWireClient(env={"CUSTOMER_MCP_LOG_LEVEL": "verbose"}) as client:
        assert client.initialize()["result"]["serverInfo"]["name"] == "customer-service"
    assert client.proc.returncode == 0
    assert "ignoring CUSTOMER_MCP_LOG_LEVEL='verbose'" in client.stderr
    assert "starting customer-service" in client.stderr  # INFO is in effect
    assert "Traceback" not in client.stderr
    assert_pure_jsonrpc(client.stdout_lines)


def test_log_level_variable_is_case_insensitive() -> None:
    result = run_cli([], env={"CUSTOMER_MCP_LOG_LEVEL": "warning"})
    assert result.returncode == 0
    assert "ignoring" not in result.stderr
    assert "starting customer-service" not in result.stderr  # INFO is below WARNING
