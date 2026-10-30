"""Shutdown behaviour: closing stdin must not cost the client any reply it is owed, and Ctrl+C
must stop the server even while the client keeps stdin open."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO, Any

import mcp_types as types
import pytest

from .wire import HANDSHAKE_VERSION, SERVER_CMD, StdioWireClient, assert_pure_jsonrpc, server_env

RUNS = 10
SLOW_SERVER = [sys.executable, str(Path(__file__).with_name("slow_server.py"))]
INTERRUPT_SERVER = [sys.executable, str(Path(__file__).with_name("interrupt_server.py"))]
REASON = "Item arrived damaged in transit"
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def request(request_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def initialize(request_id: int = 1) -> dict[str, Any]:
    client = {"name": "shutdown-test", "version": "1"}
    return request(
        request_id,
        "initialize",
        {"protocolVersion": HANDSHAKE_VERSION, "capabilities": {}, "clientInfo": client},
    )


def call(request_id: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return request(request_id, "tools/call", {"name": name, "arguments": arguments})


def encode(messages: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(m).encode() + b"\n" for m in messages)


def run_server(
    stdin: bytes | IO[bytes], args: list[str] | None = None
) -> tuple[subprocess.CompletedProcess[bytes], float]:
    """Run a server on a whole session written up front, until it exits by itself."""
    started = time.monotonic()
    result = subprocess.run(
        args or SERVER_CMD,
        input=stdin if isinstance(stdin, bytes) else None,
        stdin=None if isinstance(stdin, bytes) else stdin,
        capture_output=True,
        env=server_env(),
        cwd=Path(__file__).resolve().parent,
        timeout=60,
        check=False,
    )
    return result, time.monotonic() - started


def replies(result: subprocess.CompletedProcess[bytes]) -> dict[Any, dict[str, Any]]:
    lines = result.stdout.splitlines(keepends=True)
    assert_pure_jsonrpc(lines)
    messages = [json.loads(line) for line in lines]
    by_id = {m["id"]: m for m in messages}
    assert len(by_id) == len(messages), "a request was answered more than once"
    return by_id


def scripted_session() -> tuple[bytes, set[int]]:
    """Every kind of request: SDK-answered, tool results, business errors, validation errors."""
    messages = [
        initialize(1),
        INITIALIZED,
        request(2, "tools/list"),
        request(3, "ping"),
        request(4, "no/such"),
    ]
    for n in range(5, 45):
        match n % 5:
            case 0:
                messages.append(call(n, "get_customer_record", {"customer_id": "CUST-10001"}))
            case 1:
                refund = {"customer_id": "CUST-10003", "amount": 1.25, "reason": REASON}
                messages.append(call(n, "trigger_refund", refund))
            case 2:  # business-rule failure: an isError result
                refund = {"customer_id": "CUST-10004", "amount": 5, "reason": REASON}
                messages.append(call(n, "trigger_refund", refund))
            case 3:  # validation failure: -32602
                messages.append(
                    call(n, "trigger_refund", {"customer_id": "CUST-1", "amount": -1, "reason": "x"})
                )
            case _:
                messages.append(call(n, "get_customer_record", {"customer_id": "CUST-99999"}))
    return encode(messages), {m["id"] for m in messages if "id" in m}


def test_every_request_is_answered_when_stdin_closes_early(tmp_path: Path) -> None:
    # `customer-mcp < requests.jsonl`: the file hits EOF long before the tool calls finish. The
    # server must still answer every request it read, exactly once. Parallel runs shake out races.
    payload, expected_ids = scripted_session()
    stdin_file = tmp_path / "requests.jsonl"
    stdin_file.write_bytes(payload)

    def replay(_: int) -> subprocess.CompletedProcess[bytes]:
        with stdin_file.open("rb") as stdin:
            return run_server(stdin)[0]

    with ThreadPoolExecutor(max_workers=RUNS) as pool:
        results = list(pool.map(replay, range(RUNS)))

    for run, result in enumerate(results):
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        answered = set(replies(result))
        assert answered == expected_ids, f"run {run}: no reply to {sorted(expected_ids - answered)}"


def test_pipelined_session_over_a_pipe_is_fully_answered() -> None:
    payload, expected_ids = scripted_session()
    result, _ = run_server(payload)
    assert result.returncode == 0
    assert set(replies(result)) == expected_ids


def test_a_cancelled_request_does_not_hold_up_shutdown() -> None:
    # The refund never finishes, but the client cancelled it, so the server must not wait for it
    # (MCP forbids replying to a cancelled request). Without the cancel it would wait 30 s.
    cancel = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 7}}
    refund = call(7, "trigger_refund", {"customer_id": "CUST-10001", "amount": 1, "reason": REASON})
    result, elapsed = run_server(
        encode([initialize(), INITIALIZED, refund, cancel, request(8, "ping")]), SLOW_SERVER
    )

    assert result.returncode == 0
    assert set(replies(result)) == {1, 8}
    assert elapsed < 15, f"shutdown took {elapsed:.1f}s"
    assert b"unanswered" not in result.stderr


def test_a_request_that_never_finishes_is_bounded_and_still_answered() -> None:
    refund = call(7, "trigger_refund", {"customer_id": "CUST-10001", "amount": 1, "reason": REASON})
    result, elapsed = run_server(
        encode([initialize(), INITIALIZED, refund, request(8, "ping")]), [*SLOW_SERVER, "1"]
    )

    assert result.returncode == 0
    answered = replies(result)
    assert answered[8]["result"] == {}
    # Given up on at shutdown, but still answered: the client is told the connection closed.
    assert answered[7]["error"]["code"] == types.CONNECTION_CLOSED
    assert elapsed < 15, f"shutdown took {elapsed:.1f}s"
    assert "1 request(s) still unanswered" in result.stderr.decode()


@pytest.mark.parametrize(
    "delivery",
    [
        "simulated",
        pytest.param(
            "real-sigint",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="SIGINT cannot target one process"),
        ),
    ],
)
def test_ctrl_c_stops_the_server_while_stdin_is_still_open(delivery: str) -> None:
    # The pending stdin read cannot be cancelled; it must not keep the process alive either.
    args = [*INTERRUPT_SERVER, "--simulate"] if delivery == "simulated" else INTERRUPT_SERVER
    with StdioWireClient(args=args) as client:
        client.initialize()
        lookup = {"name": "get_customer_record", "arguments": {"customer_id": "CUST-10001"}}
        assert client.request("tools/call", lookup)["result"]["isError"] is False
        if delivery == "real-sigint":
            client.proc.send_signal(signal.SIGINT)
        started = time.monotonic()
        exit_code = client.proc.wait(timeout=15)  # stdin is still open
        elapsed = time.monotonic() - started

    assert exit_code == 130
    assert elapsed < 10, f"took {elapsed:.1f}s to stop"
    assert client.stderr.count("interrupted (Ctrl+C), server stopped") == 1
    assert "stdin closed" not in client.stderr
    assert "Traceback" not in client.stderr
    assert_pure_jsonrpc(client.stdout_lines)
