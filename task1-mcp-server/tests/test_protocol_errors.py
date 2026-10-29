"""JSON-RPC error mapping, checked byte-for-byte on the wire and through the SDK client."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import mcp_types as types
import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from customer_mcp.server import create_server
from customer_mcp.store import Customer, CustomerStore

from .wire import StdioWireClient, assert_pure_jsonrpc

PING = b'{"jsonrpc":"2.0","id":"ping-after","method":"ping"}\n'


def call(request_id: int, name: str, arguments: Any) -> bytes:
    params = {"name": name, "arguments": arguments}
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": params}).encode()


@pytest.fixture(scope="module")
def wire() -> Iterator[StdioWireClient]:
    """One server for the whole module: error paths must not change server state."""
    with StdioWireClient() as client:
        client.initialize()
        yield client
    assert_pure_jsonrpc(client.stdout_lines)


def exchange(wire: StdioWireClient, frame: bytes) -> dict[str, Any]:
    wire.send_raw(frame.rstrip(b"\n") + b"\n")
    return wire.receive()


@pytest.mark.parametrize(
    ("frame", "code", "request_id"),
    [
        pytest.param(b"{not json", types.PARSE_ERROR, None, id="not-json"),
        pytest.param(b"\xff\xfe\x00garbage", types.PARSE_ERROR, None, id="invalid-utf8"),
        pytest.param(
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"trigger_refund",'
            b'"arguments":{"customer_id":"CUST-10001","amount":NaN,"reason":"Damaged in transit"}}}',
            types.PARSE_ERROR,
            None,
            id="nan-literal",
        ),
        pytest.param(b'[{"jsonrpc":"2.0","id":2,"method":"ping"}]', types.INVALID_REQUEST, None, id="batch"),
        pytest.param(b'{"jsonrpc":"1.0","id":3,"method":"ping"}', types.INVALID_REQUEST, 3, id="jsonrpc-1.0"),
        pytest.param(
            b'{"jsonrpc":"2.0","id":4,"method":7}', types.INVALID_REQUEST, 4, id="method-not-string"
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":true,"method":"ping"}', types.INVALID_REQUEST, None, id="id-bool"
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":null,"method":"ping"}', types.INVALID_REQUEST, None, id="id-null"
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":5,"method":"no/such"}', types.METHOD_NOT_FOUND, 5, id="unknown-method"
        ),
        pytest.param(call(6, "delete_customer", {}), types.INVALID_PARAMS, 6, id="unknown-tool"),
        pytest.param(
            call(7, "trigger_refund", "not-an-object"), types.INVALID_PARAMS, 7, id="arguments-not-object"
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"arguments":{}}}',
            types.INVALID_PARAMS,
            8,
            id="tool-name-missing",
        ),
    ],
)
def test_error_codes_on_the_wire(wire: StdioWireClient, frame: bytes, code: int, request_id: object) -> None:
    response = exchange(wire, frame)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == request_id
    assert response["error"]["code"] == code
    assert "result" not in response


def test_invalid_tool_arguments_report_fields_without_echoing_values(wire: StdioWireClient) -> None:
    secret = "SSN 123-45-6789"
    response = exchange(
        wire, call(20, "trigger_refund", {"customer_id": secret, "amount": "25", "reason": "short"})
    )
    error = response["error"]
    assert error["code"] == types.INVALID_PARAMS
    assert {e["field"]: e["type"] for e in error["data"]["errors"]} == {
        "customer_id": "string_pattern_mismatch",
        "amount": "float_type",
        "reason": "string_too_short",
    }
    assert secret not in json.dumps(response)


def test_business_rule_failures_are_results_not_protocol_errors(wire: StdioWireClient) -> None:
    response = exchange(wire, call(21, "get_customer_record", {"customer_id": "CUST-99999"}))
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert response["result"]["structuredContent"]["error"]["code"] == "customer_not_found"


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(b"", id="blank-line"),
        pytest.param(b'{"jsonrpc":"2.0","id":99,"result":{}}', id="unsolicited-response"),
        pytest.param(b'{"jsonrpc":"2.0","method":"notifications/unknown"}', id="unknown-notification"),
    ],
)
def test_frames_that_must_not_be_answered(wire: StdioWireClient, frame: bytes) -> None:
    wire.send_raw(frame + b"\n")
    # Frames are read in order, so a reply to `frame` would arrive before the ping's response.
    wire.send_raw(PING)
    assert wire.receive() == {"jsonrpc": "2.0", "id": "ping-after", "result": {}}


def test_server_keeps_serving_after_errors(wire: StdioWireClient) -> None:
    response = exchange(wire, call(30, "get_customer_record", {"customer_id": "CUST-10001"}))
    assert response["result"]["isError"] is False
    assert response["result"]["structuredContent"]["customer_id"] == "CUST-10001"


class _BrokenStore(CustomerStore):
    def get(self, customer_id: str) -> Customer:
        raise RuntimeError("connection to db-internal.corp:5432 failed (password=hunter2)")


@pytest.mark.anyio
async def test_unexpected_faults_are_internal_errors_without_internals() -> None:
    async with Client(create_server(_BrokenStore([]))) as client:
        with pytest.raises(MCPError) as exc:
            await client.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
    assert exc.value.code == types.INTERNAL_ERROR
    assert exc.value.message == "Internal error"
    leaked = repr(exc.value.error)
    assert "hunter2" not in leaked
    assert "db-internal" not in leaked
    assert "Traceback" not in leaked
