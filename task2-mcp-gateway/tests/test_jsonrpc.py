"""JSON-RPC parsing: strict envelope checks, and refusal of input that parsers could read differently."""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mcp_gateway.jsonrpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    JsonRpcError,
    canonical_json,
    parse_message,
)


def parse_error(body: bytes) -> JsonRpcError:
    with pytest.raises(JsonRpcError) as exc:
        parse_message(body)
    return exc.value


def with_number(number: str) -> bytes:
    """A compact (so already canonical) request whose params hold one number, written as given."""
    return b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"n":' + number.encode() + b"}}"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}', id="request"),
        pytest.param(b'{"jsonrpc":"2.0","id":"a-1","method":"ping","params":{}}', id="string-id"),
        pytest.param(b'{"jsonrpc":"2.0","method":"notifications/initialized"}', id="notification"),
        pytest.param(b'{"jsonrpc":"2.0","id":"s-1","result":{"action":"accept"}}', id="client-response"),
        pytest.param(b'{"jsonrpc":"2.0","id":"s-1","error":{"code":-1,"message":"no"}}', id="client-error"),
        pytest.param(
            b'{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"x"}}', id="error-null-id"
        ),
    ],
)
def test_valid_messages_are_accepted(body: bytes) -> None:
    message = parse_message(body)
    assert message.payload == json.loads(body)


def test_canonical_form_is_compact_utf8_of_the_parsed_object() -> None:
    body = (
        b'{ "jsonrpc" : "2.0", "id" : 1, "method" : "tools/call",\n "params" : '
        b'{"name" : "get\\u005fweather", "arguments": {"city": "Z\\u00fcrich", "n": 1E2}} }'
    )
    message = parse_message(body)
    expected = (
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
        '{"name":"get_weather","arguments":{"city":"Zürich","n":1E2}}}'
    )
    assert message.canonical == expected.encode()
    assert message.method == "tools/call"
    assert message.id == 1
    assert message.params == {"name": "get_weather", "arguments": {"city": "Zürich", "n": 100.0}}


@pytest.mark.parametrize(
    "number",
    [
        "12345678901234567890.5",  # float() would round it to 1.2345678901234567e+19
        "1E2",  # float() would write 100.0
        "1.10",
        "0.1000000000000000055511151231257827",
        "-0.0",
        "5e-324",
        "1.5E+300",
        "-2.5e-7",
        "123456789012345678901234567890",  # an integer: Python ints are exact
    ],
)
def test_numbers_are_forwarded_exactly_as_written(number: str) -> None:
    message = parse_message(with_number(number))
    assert message.canonical == with_number(number)
    assert message.params == {"n": json.loads(number)}  # what the gateway itself reads


def test_negative_zero_integer_is_forwarded_as_zero() -> None:
    assert parse_message(with_number("-0")).canonical == with_number("0")


@pytest.mark.parametrize("number", ["1.\u0661", "\u0661.5", "1e\u0662", "1.5\uff10"])
def test_numbers_with_non_ascii_digits_are_a_parse_error(number: str) -> None:
    assert parse_error(with_number(number)).code == PARSE_ERROR


_JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=20,
)


@settings(max_examples=300, deadline=None)
@given(_JSON_VALUES)
def test_canonical_form_matches_compact_json_dumps(value: Any) -> None:
    # For input written by Python's own json module the canonical form is exactly what
    # json.dumps writes: only number lexemes from other writers are kept differently.
    payload = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"value": value}}
    message = parse_message(json.dumps(payload).encode())
    assert message.canonical == json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    assert json.loads(message.canonical) == json.loads(json.dumps(payload))


_NUMBER_LEXEMES = st.from_regex(
    r"-?(0|[1-9][0-9]{0,25})(\.[0-9]{1,25})?([eE][-+]?[0-9]{1,3})?", fullmatch=True
)


@settings(max_examples=300, deadline=None)
@given(_NUMBER_LEXEMES)
def test_every_finite_number_keeps_its_text(number: str) -> None:
    try:
        message = parse_message(with_number(number))
    except JsonRpcError as exc:
        assert exc.code == PARSE_ERROR
        assert not math.isfinite(float(number))  # only out-of-range numbers are refused
        return
    is_integer = not any(mark in number for mark in ".eE")
    assert message.canonical == with_number(str(int(number)) if is_integer else number)


def test_canonical_json_is_not_limited_by_the_recursion_limit() -> None:
    depth = sys.getrecursionlimit() * 5
    value: Any = []
    for _ in range(depth - 1):
        value = [value]
    assert canonical_json(value) == "[" * depth + "]" * depth


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"{", id="truncated"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping"} trailing', id="trailing-data"),
        pytest.param(
            b'{"jsonrpc":"2.0","id":1,"method":"ping","method":"tools/call"}', id="duplicate-method"
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_weather","name":"admin_x"}}',
            id="duplicate-tool-name",
        ),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"id":2,"method":"ping"}', id="duplicate-id"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":NaN}}', id="NaN"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":-Infinity}}', id="Infinity"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":1e999}}', id="float-overflow"),
        pytest.param(
            b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":' + b"9" * 5000 + b"}}", id="huge-int"
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_\\ud800x"}}',
            id="lone-surrogate",
        ),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"pi\xffng"}', id="invalid-utf8"),
        pytest.param(b'\xef\xbb\xbf{"jsonrpc":"2.0","id":1,"method":"ping"}', id="utf8-bom"),
        pytest.param('{"jsonrpc":"2.0","id":1,"method":"ping"}'.encode("utf-16"), id="utf16"),
        pytest.param(b"[" * 100_000 + b"]" * 100_000, id="deep-nesting"),
    ],
)
def test_ambiguous_or_malformed_json_is_a_parse_error(body: bytes) -> None:
    error = parse_error(body)
    assert error.code == PARSE_ERROR
    assert error.rpc_id is None
    assert error.http_status == 400


@pytest.mark.parametrize(
    ("body", "rpc_id"),
    [
        pytest.param(b'[{"jsonrpc":"2.0","id":1,"method":"tools/list"}]', None, id="batch"),
        pytest.param(b"[]", None, id="empty-batch"),
        pytest.param(b'"tools/call"', None, id="string"),
        pytest.param(b"null", None, id="null"),
        pytest.param(b'{"id":1,"method":"ping"}', 1, id="missing-jsonrpc"),
        pytest.param(b'{"jsonrpc":"1.0","id":1,"method":"ping"}', 1, id="jsonrpc-1.0"),
        pytest.param(b'{"jsonrpc":2.0,"id":1,"method":"ping"}', 1, id="jsonrpc-number"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":""}', 1, id="empty-method"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":["tools/call"]}', 1, id="method-array"),
        pytest.param(b'{"jsonrpc":"2.0","id":null,"method":"ping"}', None, id="null-id"),
        pytest.param(b'{"jsonrpc":"2.0","id":1.5,"method":"ping"}', None, id="float-id"),
        pytest.param(b'{"jsonrpc":"2.0","id":true,"method":"ping"}', None, id="bool-id"),
        pytest.param(b'{"jsonrpc":"2.0","id":{"a":1},"method":"ping"}', None, id="object-id"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","params":"x"}', 1, id="string-params"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","params":null}', 1, id="null-params"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"ping","Method":"tools/call"}', 1, id="extra-member"),
        pytest.param(b'{"jsonrpc":"2.0","id":1}', 1, id="neither-request-nor-response"),
        pytest.param(b'{"jsonrpc":"2.0","id":1,"result":{},"error":{}}', 1, id="result-and-error"),
        pytest.param(b'{"jsonrpc":"2.0","result":{}}', None, id="response-without-id"),
        pytest.param(b'{"jsonrpc":"2.0","id":null,"result":{}}', None, id="result-null-id"),
    ],
)
def test_invalid_envelopes_are_invalid_requests(body: bytes, rpc_id: int | None) -> None:
    error = parse_error(body)
    assert error.code == INVALID_REQUEST
    assert error.rpc_id == rpc_id


def test_positional_params_are_invalid_params() -> None:
    error = parse_error(b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":["admin_reset_key"]}')
    assert error.code == INVALID_PARAMS
    assert error.rpc_id == 3


def test_parse_errors_do_not_echo_the_input() -> None:
    error = parse_error(b'{"jsonrpc":"2.0","id":1,"method":"x","method":"secret-value"}')
    assert "secret-value" not in error.message
