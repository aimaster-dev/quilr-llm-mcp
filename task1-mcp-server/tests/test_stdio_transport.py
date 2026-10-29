"""The server speaks newline-delimited JSON-RPC on stdout and logs only to stderr."""

from __future__ import annotations

from customer_mcp import SERVER_NAME

from .wire import StdioWireClient, assert_pure_jsonrpc


def test_initialize_handshake_over_stdio() -> None:
    with StdioWireClient() as client:
        response = client.initialize()
        assert response["result"]["serverInfo"]["name"] == SERVER_NAME
        assert "tools" in response["result"]["capabilities"]
        assert client.request("ping")["result"] == {}

    assert_pure_jsonrpc(client.stdout_lines)


def test_logs_are_written_to_stderr_only() -> None:
    with StdioWireClient(env={"CUSTOMER_MCP_LOG_LEVEL": "DEBUG"}) as client:
        client.initialize()
        client.request("tools/list")
    exit_code = client.proc.returncode

    assert exit_code == 0
    assert f"starting {SERVER_NAME}" in client.stderr
    assert "server stopped" in client.stderr
    assert_pure_jsonrpc(client.stdout_lines)
    assert all(b"starting" not in line for line in client.stdout_lines)


def test_server_exits_cleanly_when_stdin_closes() -> None:
    client = StdioWireClient()
    client.initialize()
    assert client.close() == 0
