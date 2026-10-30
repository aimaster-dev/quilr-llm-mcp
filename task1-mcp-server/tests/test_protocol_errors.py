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
        # The next four used to get no reply: the SDK's parser rejects the first two (json.loads
        # does not) and reads the last two as a response and a request. Each id is still echoed.
        pytest.param(
            b'{"jsonrpc":"2.0","id":42,"method":"tools/call","params":{"name":"get_customer_record",'
            b'"arguments":{"customer_id":"\\ud800"}}}',
            types.PARSE_ERROR,
            42,
            id="lone-surrogate-escape",
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":43,"method":"tools/call","params":{"name":"get_customer_record",'
            b'"arguments":{"customer_id":' + b"[" * 250 + b"]" * 250 + b"}}}",
            types.PARSE_ERROR,
            43,
            id="nesting-250",
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":77,"method":"ping","error":{"code":1,"message":"x"}}',
            types.INVALID_REQUEST,
            77,
            id="method-and-error",
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":78,"method":"ping","result":{}}',
            types.INVALID_REQUEST,
            78,
            id="method-and-result",
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"trigger_refund",'
            b'"arguments":{"customer_id":"CUST-10001","amount":NaN,"reason":"Damaged in transit"}}}',
            types.PARSE_ERROR,
            1,
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


def test_unexpected_argument_names_are_redacted_unless_identifiers(wire: StdioWireClient) -> None:
    secret = "SSN 123-45-6789"
    response = exchange(
        wire, call(24, "get_customer_record", {"customer_id": "CUST-10001", secret: 1, "customerId": "x"})
    )
    errors = response["error"]["data"]["errors"]
    # A typo is worth naming; a key that carries data is not echoed.
    assert {(e["field"], e["type"]) for e in errors} == {
        ("(redacted)", "extra_forbidden"),
        ("customerId", "extra_forbidden"),
    }
    assert "123-45-6789" not in json.dumps(response)


@pytest.mark.parametrize(
    ("arguments_or_name", "field", "error_type"),
    [
        pytest.param({"arguments": "CUST-10001"}, "arguments", "dict_type", id="arguments-string"),
        pytest.param({"arguments": ["CUST-10001"]}, "arguments", "dict_type", id="arguments-array"),
        pytest.param({"name": 5}, "name", "string_type", id="name-not-string"),
    ],
)
def test_malformed_tools_call_params_get_field_level_details(
    wire: StdioWireClient, arguments_or_name: dict[str, Any], field: str, error_type: str
) -> None:
    # The SDK validates these before the tool registry sees them and answers with `data: ""`.
    params = {"name": "get_customer_record", "arguments": {}, **arguments_or_name}
    frame = json.dumps({"jsonrpc": "2.0", "id": 25, "method": "tools/call", "params": params}).encode()
    error = exchange(wire, frame)["error"]
    assert error["code"] == types.INVALID_PARAMS
    assert {"field": field, "type": error_type} in [
        {"field": e["field"], "type": e["type"]} for e in error["data"]["errors"]
    ]
    assert "CUST-10001" not in json.dumps(error)


@pytest.mark.parametrize(
    ("method", "echoed"),
    [
        pytest.param("no/such", True, id="plain-name"),
        pytest.param("resources/list", True, id="spec-method-not-served"),
        pytest.param("A" * 100_000, False, id="100kb-name"),
        pytest.param("tools/call\u001b[31m", False, id="control-characters"),
        pytest.param("SSN 123-45-6789", False, id="free-text"),
    ],
)
def test_unknown_method_name_is_echoed_only_when_it_looks_like_one(
    wire: StdioWireClient, method: str, echoed: bool
) -> None:
    response = exchange(wire, json.dumps({"jsonrpc": "2.0", "id": 26, "method": method}).encode())
    error = response["error"]
    assert error["code"] == types.METHOD_NOT_FOUND
    assert error["message"] == "Method not found"
    if echoed:
        assert error["data"] == method
    else:
        assert "data" not in error
    assert len(json.dumps(response)) < 300


@pytest.mark.parametrize(
    ("name", "echoed"),
    [
        pytest.param("delete_customer", True, id="plain-name"),
        pytest.param("get_customer_record\u001b[31m", False, id="control-characters"),
        pytest.param("T" * 129, False, id="too-long"),
        pytest.param("SSN 123-45-6789", False, id="free-text"),
    ],
)
def test_unknown_tool_name_is_echoed_only_when_it_looks_like_one(
    wire: StdioWireClient, name: str, echoed: bool
) -> None:
    error = exchange(wire, call(27, name, {}))["error"]
    assert error["code"] == types.INVALID_PARAMS
    assert error["message"] == "Unknown tool"
    assert error.get("data") == ({"tool": name} if echoed else None)


def test_invalid_utf8_inside_a_string_is_rejected_not_replaced(wire: StdioWireClient) -> None:
    # Decoding with errors="replace" once issued this refund with reason "Damaged \ufffd\ufffd in transit".
    response = exchange(
        wire,
        b'{"jsonrpc":"2.0","id":22,"method":"tools/call","params":{"name":"trigger_refund","arguments":'
        b'{"customer_id":"CUST-10003","amount":1.0,"reason":"Damaged \xff\xfe in transit"}}}',
    )
    assert response["id"] == 22
    assert response["error"]["code"] == types.PARSE_ERROR
    record = exchange(wire, call(23, "get_customer_record", {"customer_id": "CUST-10003"}))
    assert record["result"]["structuredContent"]["refundable_balance"] == "9800.00"


def test_frames_are_split_on_lf_only(wire: StdioWireClient) -> None:
    # A bare CR between tokens is JSON whitespace: one frame, one normal reply.
    response = exchange(wire, b'{"jsonrpc":"2.0",\r"id":"cr","method":"ping"}')
    assert response == {"jsonrpc": "2.0", "id": "cr", "result": {}}
    # A bare CR inside a string is an invalid control character: exactly one parse error, not two.
    wire.send_raw(b'{"jsonrpc":"2.0","id":"cr2","method":"pi\rng"}\n')
    wire.send_raw(PING)
    assert wire.receive()["error"]["code"] == types.PARSE_ERROR
    assert wire.receive()["id"] == "ping-after"
    # CRLF-terminated frames keep working.
    wire.send_raw(b'{"jsonrpc":"2.0","id":"crlf","method":"ping"}\r\n')
    assert wire.receive() == {"jsonrpc": "2.0", "id": "crlf", "result": {}}


def test_business_rule_failures_are_results_not_protocol_errors(wire: StdioWireClient) -> None:
    response = exchange(wire, call(21, "get_customer_record", {"customer_id": "CUST-99999"}))
    assert "error" not in response
    assert response["result"]["isError"] is True
    # No structuredContent (it would violate the outputSchema); the error is JSON in the text.
    assert "structuredContent" not in response["result"]
    assert json.loads(response["result"]["content"][0]["text"])["error"]["code"] == "customer_not_found"


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(b"", id="blank-line"),
        pytest.param(b'{"jsonrpc":"2.0","id":99,"result":{}}', id="unsolicited-response"),
        pytest.param(b'{"jsonrpc":"2.0","result":{}}', id="malformed-response"),
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
