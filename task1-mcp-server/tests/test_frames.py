"""Unit tests for inbound frame validation (`check_frame`), the layer in front of the SDK."""

from __future__ import annotations

import mcp_types as types
import pytest

from customer_mcp.stdio import DROP, FORWARD, check_frame


@pytest.mark.parametrize(
    "line",
    [
        '{"jsonrpc":"2.0","id":1,"method":"ping"}\n',
        '{"jsonrpc":"2.0","id":0,"method":"ping"}',
        '{"jsonrpc":"2.0","id":-3,"method":"ping"}',
        '{"jsonrpc":"2.0","id":"req-abc","method":"tools/list","params":{}}',
        '{"jsonrpc":"2.0","method":"notifications/initialized"}',
        # Responses to server-initiated requests (e.g. sampling) are the SDK's business.
        '{"jsonrpc":"2.0","id":5,"result":{}}',
        '{"jsonrpc":"2.0","id":5,"error":{"code":-1,"message":"declined"}}',
    ],
)
def test_forwards_well_formed_messages(line: str) -> None:
    assert check_frame(line) == FORWARD


@pytest.mark.parametrize("line", ["", "\n", "   \r\n", "\t\n"])
def test_drops_blank_lines(line: str) -> None:
    assert check_frame(line) == DROP


@pytest.mark.parametrize(
    "line",
    [
        "{not json\n",
        '{"jsonrpc":"2.0","id":1,"method":"ping"',  # truncated
        "�� garbage",  # what invalid UTF-8 decodes to
        # Python's json module accepts these literals, JSON (and the SDK's parser) does not.
        "NaN",
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"arguments":{"amount":Infinity}}}',
        "[" * 100_000,  # nesting deep enough to exhaust the recursion limit
    ],
)
def test_unparseable_lines_are_parse_errors(line: str) -> None:
    check = check_frame(line)
    assert check.forward is False
    assert check.error is not None
    assert check.error.id is None
    assert check.error.error.code == types.PARSE_ERROR
    assert check.error.error.message == "Parse error"


@pytest.mark.parametrize(
    ("line", "expected_id", "reason"),
    [
        ("[]", None, "Batch requests are not supported"),
        ('[{"jsonrpc":"2.0","id":1,"method":"ping"}]', None, "Batch requests are not supported"),
        ('"ping"', None, "Message must be a JSON object"),
        ("42", None, "Message must be a JSON object"),
        ("null", None, "Message must be a JSON object"),
        ("{}", None, '"jsonrpc" must be exactly "2.0"'),
        ('{"jsonrpc":"1.0","id":7,"method":"ping"}', 7, '"jsonrpc" must be exactly "2.0"'),
        ('{"jsonrpc":2.0,"id":7,"method":"ping"}', 7, '"jsonrpc" must be exactly "2.0"'),
        ('{"id":"a","method":"ping"}', "a", '"jsonrpc" must be exactly "2.0"'),
        ('{"jsonrpc":"2.0","id":8,"method":42}', 8, '"method" must be a string'),
        ('{"jsonrpc":"2.0","id":8,"method":null}', 8, '"method" must be a string'),
        ('{"jsonrpc":"2.0","id":9}', 9, '"method" is required'),
        # The SDK would read these as notifications and never answer them.
        ('{"jsonrpc":"2.0","id":true,"method":"ping"}', None, '"id" must be a string or an integer'),
        ('{"jsonrpc":"2.0","id":null,"method":"ping"}', None, '"id" must be a string or an integer'),
        ('{"jsonrpc":"2.0","id":1.5,"method":"ping"}', None, '"id" must be a string or an integer'),
        ('{"jsonrpc":"2.0","id":[1],"method":"ping"}', None, '"id" must be a string or an integer'),
        ('{"jsonrpc":"2.0","id":{"n":1},"method":"ping"}', None, '"id" must be a string or an integer'),
        (
            '{"jsonrpc":"2.0","id":"x","method":"tools/call","params":[1,2]}',
            "x",
            '"params" must be an object',
        ),
        ('{"jsonrpc":"2.0","method":"tools/call","params":"str"}', None, '"params" must be an object'),
    ],
)
def test_invalid_envelopes_are_invalid_requests(line: str, expected_id: object, reason: str) -> None:
    check = check_frame(line)
    assert check.forward is False
    assert check.error is not None
    assert check.error.id == expected_id
    assert check.error.error.code == types.INVALID_REQUEST
    assert check.error.error.message == "Invalid Request"
    assert check.error.error.data == reason


def test_error_data_never_echoes_input() -> None:
    secret = "4111-1111-1111-1111"
    check = check_frame('{"jsonrpc":"1.0","id":1,"method":"' + secret + '","params":{"ssn":"123-45-6789"}}')
    assert check.error is not None
    dumped = check.error.model_dump_json()
    assert secret not in dumped
    assert "123-45-6789" not in dumped
