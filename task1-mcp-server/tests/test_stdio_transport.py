"""The server speaks newline-delimited JSON-RPC on stdout and logs only to stderr."""

from __future__ import annotations

import sys
from pathlib import Path

import mcp_types as types
import pytest
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError

from customer_mcp import SERVER_NAME

from .wire import StdioWireClient, assert_pure_jsonrpc

NOISY_SERVER = [sys.executable, str(Path(__file__).with_name("noisy_server.py"))]


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


def test_stray_output_never_reaches_the_wire() -> None:
    with StdioWireClient(args=NOISY_SERVER) as client:
        client.initialize()
        response = client.request(
            "tools/call", {"name": "get_customer_record", "arguments": {"customer_id": "CUST-10001"}}
        )
        assert response["result"]["structuredContent"]["name"] == "Ada Lovelace"

    assert_pure_jsonrpc(client.stdout_lines)
    for noise in ("stray print() from a tool handler", "raw write to fd 1", "child stdout"):
        assert noise in client.stderr
        assert all(noise.encode() not in line for line in client.stdout_lines)
    # The child inherited fd 0 too, but it points at the null device, not the wire.
    assert "child stdin: ''" in client.stderr


@pytest.mark.anyio
async def test_official_sdk_client_over_stdio() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "customer_mcp"])
    async with Client(params) as client:
        assert {tool.name for tool in (await client.list_tools()).tools} == {
            "get_customer_record",
            "trigger_refund",
        }

        record = await client.call_tool("get_customer_record", {"customer_id": "CUST-10003"})
        assert record.is_error is False
        assert record.structured_content["name"] == "Grace Hopper"

        receipt = await client.call_tool(
            "trigger_refund", {"customer_id": "CUST-10003", "amount": 99.99, "reason": "Late delivery refund"}
        )
        assert receipt.structured_content["remaining_refundable_balance"] == "9700.01"

        with pytest.raises(MCPError) as exc:
            await client.call_tool(
                "trigger_refund", {"customer_id": "CUST-10003", "amount": -1, "reason": "x"}
            )
        assert exc.value.code == types.INVALID_PARAMS
