"""The server speaks newline-delimited JSON-RPC on stdout and logs only to stderr."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mcp_types as types
import pytest
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError

from customer_mcp import SERVER_NAME

from .wire import SERVER_CMD, StdioWireClient, assert_pure_jsonrpc, server_env

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


def test_tools_before_initialize_get_the_sdks_invalid_params() -> None:
    # Documented SDK behaviour, passed through unchanged: only `ping` is served before the handshake.
    with StdioWireClient() as client:
        assert client.request("tools/list")["error"] == {
            "code": types.INVALID_PARAMS,
            "message": "Invalid request parameters",
            "data": "",
        }
        assert client.request("ping")["result"] == {}
        assert client.request("no/such")["error"]["code"] == types.METHOD_NOT_FOUND
    assert_pure_jsonrpc(client.stdout_lines)


def test_batches_are_rejected_even_at_2025_03_26() -> None:
    # 2025-03-26 required servers to accept batches; neither the SDK nor this server does (README).
    with StdioWireClient() as client:
        init = client.request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        )
        assert init["result"]["protocolVersion"] == "2025-03-26"
        client.notify("notifications/initialized")
        client.send(
            [
                {"jsonrpc": "2.0", "id": "b1", "method": "ping"},
                {"jsonrpc": "2.0", "id": "b2", "method": "ping"},
            ]
        )
        assert client.receive() == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": types.INVALID_REQUEST,
                "message": "Invalid Request",
                "data": "Batch requests are not supported",
            },
        }
    assert_pure_jsonrpc(client.stdout_lines)


def test_server_exits_cleanly_when_stdin_closes() -> None:
    client = StdioWireClient()
    client.initialize()
    assert client.close() == 0


PRE_MAIN_NOISE = ("pre-main print()", "pre-main sys.__stdout__ write")
NOISE = (
    "startup print()",
    "startup library log",
    "stray print() from a tool handler",
    "raw write to fd 1",
    "unflushed sys.__stdout__ write while serving",
    "library log while serving",
    "child stdout",
    "atexit print()",
    "atexit write to fd 1",
    "atexit sys.__stdout__ write",
)


@pytest.mark.parametrize("unbuffered", [False, True], ids=["buffered-like-a-host", "unbuffered"])
def test_stray_output_never_reaches_the_wire(unbuffered: bool) -> None:
    # Hosts launch servers with a fully buffered stdout pipe, so the default run is the one that
    # catches output flushed onto the wire late (at exit); the unbuffered run catches it early.
    with StdioWireClient(args=NOISY_SERVER, unbuffered=unbuffered) as client:
        client.initialize()
        for customer_id in ("CUST-10001", "CUST-10002"):
            response = client.request(
                "tools/call", {"name": "get_customer_record", "arguments": {"customer_id": customer_id}}
            )
            assert response["result"]["isError"] is False
    assert client.proc.returncode == 0

    assert_pure_jsonrpc(client.stdout_lines)
    wire = b"".join(client.stdout_lines)
    for noise in NOISE if unbuffered else (*PRE_MAIN_NOISE, *NOISE):
        assert noise in client.stderr, f"{noise!r} went missing"
        assert noise.encode() not in wire, f"{noise!r} reached the wire"
    # The child inherited fd 0 too, but it points at the null device, not the wire.
    assert "child stdin: ''" in client.stderr


# Python sets sys.stdin to None when fd 0 is closed at startup (`customer-mcp <&-`); this wrapper
# reproduces that state on every platform.
STDIN_CLOSED = [
    sys.executable,
    "-c",
    "import os, sys; os.close(0); sys.stdin = None; from customer_mcp.__main__ import main; sys.exit(main())",
]


@pytest.mark.parametrize("args", [SERVER_CMD, STDIN_CLOSED], ids=["stdin-at-eof", "stdin-closed"])
def test_server_exits_cleanly_without_input(args: list[str]) -> None:
    result = subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=server_env(),
        cwd=Path(__file__).resolve().parent,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b""
    assert b"Traceback" not in result.stderr
    if args is STDIN_CLOSED:
        assert b"stdin is not open" in result.stderr


def test_server_refuses_to_start_without_stdout() -> None:
    # With fd 1 closed there is nowhere to answer; say so on stderr instead of serving into the void.
    stdout_closed = [
        sys.executable,
        "-c",
        "import os, sys; os.close(1); sys.stdout = None; "
        "from customer_mcp.__main__ import main; sys.exit(main())",
    ]
    result = subprocess.run(
        stdout_closed,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=server_env(),
        cwd=Path(__file__).resolve().parent,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1
    assert b"stdout is not open" in result.stderr
    assert b"Traceback" not in result.stderr


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
