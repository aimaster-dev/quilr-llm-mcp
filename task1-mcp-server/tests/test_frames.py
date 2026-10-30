"""Unit tests for inbound framing: `LinePump` splits stdin into lines and `check_frame` validates
each one, the layer in front of the SDK."""

from __future__ import annotations

import io
import os
import threading

import anyio
import mcp_types as types
import pytest

from customer_mcp.stdio import DROP, LinePump, check_frame


def nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def call_line(arguments: str, request_id: str = "1") -> str:
    return (
        '{"jsonrpc":"2.0","id":' + request_id + ',"method":"tools/call",'
        '"params":{"name":"get_customer_record","arguments":' + arguments + "}}"
    )


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ('{"jsonrpc":"2.0","id":1,"method":"ping"}\n', types.JSONRPCRequest),
        ('{"jsonrpc":"2.0","id":0,"method":"ping"}', types.JSONRPCRequest),
        ('{"jsonrpc":"2.0","id":-3,"method":"ping"}', types.JSONRPCRequest),
        ('{"jsonrpc":"2.0","id":18446744073709551616,"method":"ping"}', types.JSONRPCRequest),
        ('{"jsonrpc":"2.0","id":"req-abc","method":"tools/list","params":{}}', types.JSONRPCRequest),
        ('{"jsonrpc":"2.0","id":1,"method":"ping"}\r\n', types.JSONRPCRequest),  # CRLF terminator
        ('{"jsonrpc":"2.0",\r"id":1,"method":"ping"}\n', types.JSONRPCRequest),  # CR is JSON whitespace
        (call_line('{"customer_id":"\\ud83d\\ude00"}'), types.JSONRPCRequest),  # a valid surrogate pair
        (call_line('{"customer_id":' + nested(150) + "}"), types.JSONRPCRequest),
        ('{"jsonrpc":"2.0","method":"notifications/initialized"}', types.JSONRPCNotification),
        # Responses to server-initiated requests (e.g. sampling) are the SDK's business.
        ('{"jsonrpc":"2.0","id":5,"result":{}}', types.JSONRPCResponse),
        ('{"jsonrpc":"2.0","id":5,"error":{"code":-1,"message":"declined"}}', types.JSONRPCError),
        ('{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}', types.JSONRPCError),
    ],
)
def test_forwards_well_formed_messages_as_the_sdk_parses_them(line: str, kind: type) -> None:
    check = check_frame(line)
    assert check.forward is True
    assert isinstance(check.message, kind)
    assert check_frame(line.encode()).forward is True


@pytest.mark.parametrize("line", ["", "\n", "   \r\n", "\t\n", b"\r\n"])
def test_drops_blank_lines(line: str | bytes) -> None:
    assert check_frame(line) == DROP


@pytest.mark.parametrize(
    ("line", "expected_id"),
    [
        ("{not json\n", None),
        ('{"jsonrpc":"2.0","id":1,"method":"ping"', None),  # truncated
        ("�� garbage", None),  # what invalid UTF-8 used to decode to
        # Python's json module accepts these literals, JSON (and the SDK's parser) does not.
        ("NaN", None),
        ('{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"arguments":{"amount":Infinity}}}', 1),
        ("[" * 100_000, None),
        # json.loads accepts these; the SDK's parser rejects them, so they used to get no reply.
        (call_line('{"customer_id":"\\ud800"}', "42"), 42),  # lone surrogate escape
        (call_line('{"customer_id":"\\udc00abc"}', '"s"'), "s"),
        (call_line('{"customer_id":' + nested(250) + "}", "43"), 43),  # deeper than the SDK allows
        (call_line('{"customer_id":' + nested(990) + "}"), 1),
        ('{"jsonrpc":"2.0","id":1,"method":"ping"} xx', None),  # trailing data
        ('{"jsonrpc":"2.0","id":1,"method":"ping"}{"jsonrpc":"2.0","id":2,"method":"ping"}', None),
        ('﻿{"jsonrpc":"2.0","id":1,"method":"ping"}', None),  # a BOM is not JSON whitespace
        (call_line('{"customer_id":"CUST-\r10001"}', "5"), 5),  # a raw CR inside a string
        ("　", None),  # Unicode whitespace is not JSON whitespace
        # The id is salvaged only when it is itself intact.
        ('{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}', None),
        ('{"jsonrpc":"2.0","id":true,"method":"ping","params":{"a":"\\ud800"}}', None),
        ('{"jsonrpc":"2.0","id":9,"result":{"a":"\\ud800"}}', None),  # responses are never answered
    ],
)
def test_unparseable_lines_are_parse_errors(line: str, expected_id: object) -> None:
    check = check_frame(line)
    assert check.forward is False
    assert check.error is not None
    # JSON-RPC wants `id: null` only when the id cannot be detected: a client waiting on request
    # 42 is told about it whenever a lenient parse can still find the id.
    assert check.error.id == expected_id
    assert check.error.error.code == types.PARSE_ERROR
    assert check.error.error.message == "Parse error"


@pytest.mark.parametrize(
    ("raw", "expected_id"),
    [
        (b"\xff\xfe garbage", None),
        # Invalid UTF-8 *inside a string* must not be replaced with U+FFFD and processed.
        (
            b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"trigger_refund","arguments":'
            b'{"customer_id":"CUST-10003","amount":1,"reason":"Damaged \xff\xfe in transit"}}}\n',
            3,
        ),
        (b'{"jsonrpc":"2.0","id":1,"method":"\xc0\xafping"}', 1),  # overlong encoding
        (b'{"jsonrpc":"2.0","id":"a","method":"\xed\xa0\x80"}', "a"),  # an encoded surrogate
        (b'{"jsonrpc":"2.0","id":1,"method":"ping\xe2\x82"}', 1),  # truncated sequence
        (b'{"jsonrpc":"2.0","id":"r\xff","method":"ping"}', None),  # the id itself is damaged
    ],
)
def test_invalid_utf8_is_a_parse_error(raw: bytes, expected_id: object) -> None:
    check = check_frame(raw)
    assert check.error is not None
    assert check.error.id == expected_id
    assert check.error.error.code == types.PARSE_ERROR
    assert check.error.error.data == "Invalid UTF-8"


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
        ('{"jsonrpc":"2.0","id":1.0,"method":"ping"}', None, '"id" must be a string or an integer'),
        ('{"jsonrpc":"2.0","id":[1],"method":"ping"}', None, '"id" must be a string or an integer'),
        ('{"jsonrpc":"2.0","id":{"n":1},"method":"ping"}', None, '"id" must be a string or an integer'),
        # The SDK reads the first as a *response* and the second as a request; neither is valid.
        (
            '{"jsonrpc":"2.0","id":77,"method":"ping","error":{"code":1,"message":"x"}}',
            77,
            '"method" cannot be combined with "result" or "error"',
        ),
        (
            '{"jsonrpc":"2.0","id":78,"method":"ping","result":{}}',
            78,
            '"method" cannot be combined with "result" or "error"',
        ),
        (
            '{"jsonrpc":"2.0","id":"x","method":"tools/call","params":[1,2]}',
            "x",
            '"params" must be an object',
        ),
        ('{"jsonrpc":"2.0","id":"pn","method":"ping","params":null}', "pn", '"params" must be an object'),
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


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        ('{"jsonrpc":"2.0","result":{}}', '"id" is required'),
        ('{"jsonrpc":"1.0","id":1,"result":{}}', '"jsonrpc" must be exactly "2.0"'),
        ('{"jsonrpc":"2.0","id":1,"result":{},"error":{"code":1,"message":"x"}}', "not both"),
        ('{"jsonrpc":"2.0","id":null,"result":{}}', '"id" must be a string or an integer'),
        ('{"jsonrpc":"2.0","id":true,"error":{"code":1,"message":"x"}}', '"id" must be a string'),
        ('{"jsonrpc":"2.0","id":1,"error":{"code":"x","message":"x"}}', "not a valid JSON-RPC response"),
        ('{"jsonrpc":"2.0","id":1,"result":[]}', "not a valid JSON-RPC response"),
    ],
)
def test_malformed_responses_are_dropped_not_answered(line: str, problem: str) -> None:
    # JSON-RPC never answers a response, and its id belongs to the server's id space: echoing it
    # in an error would collide with the client's own pending request ids.
    check = check_frame(line)
    assert check.forward is False
    assert check.error is None
    assert check.problem is not None
    assert problem in check.problem


# Lines an auditor found the SDK's parser rejects (or misreads) while `json.loads` accepted them,
# plus their neighbours. Whatever check_frame forwards must parse in the SDK as the same kind of
# message; everything else must be answered (or, for a response, dropped).
DIFFERENTIAL_CORPUS = [
    call_line('{"customer_id":"\\ud800"}'),
    call_line('{"customer_id":"\\udc00abc"}'),
    *(call_line('{"customer_id":' + nested(depth) + "}") for depth in (150, 190, 195, 200, 250, 600, 990)),
    call_line('{"customer_id":1e400}'),
    call_line('{"customer_id":1' + "0" * 400 + "}"),
    call_line('{"customer_id":1' + "0" * 5000 + "}"),
    *(call_line('{"customer_id":"CUST-10001"}', rid) for rid in (str(2**64), "1" + "0" * 5000, "1e2", '""')),
    '{"jsonrpc":"2.0","id":true,"id":1,"method":"ping"}',
    '{"jsonrpc":"2.0","id":1,"method":5,"method":"ping"}',
    '{"jsonrpc":"2.0","id":1,"method":"ping","result":{}}',
    '{"jsonrpc":"2.0","id":1,"method":"ping","error":{"code":1,"message":"x"}}',
    '{"jsonrpc":"2.0","id":1,"method":"ping","foo":1}',
    '{"jsonrpc":"2.0","id":1,"method":"ping","params":null}',
    '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"_meta":5}}',
    '{"jsonrpc":"2.0","id":1,"method":"pi\\u0000ng"}',
    '{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":true}}',
    '{"jsonrpc":"2.0","result":{}}',
    '{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"x"}}',
    '{"jsonrpc":"1.0","id":1,"result":{}}',
    '{"jsonrpc":"2.0","id":1,"result":{},"error":{"code":1,"message":"x"}}',
]


@pytest.mark.parametrize("line", DIFFERENTIAL_CORPUS)
def test_nothing_is_forwarded_that_the_sdk_cannot_parse(line: str) -> None:
    check = check_frame(line + "\n")
    if not check.forward:
        assert check.error is not None or check.problem is not None
        return
    parsed = types.jsonrpc_message_adapter.validate_json(line + "\n", by_name=False)
    assert type(parsed) is type(check.message)
    claims_request = '"method"' in line
    if claims_request:
        assert isinstance(parsed, types.JSONRPCRequest | types.JSONRPCNotification)
    else:
        assert isinstance(parsed, types.JSONRPCResponse | types.JSONRPCError)


def test_error_data_never_echoes_input() -> None:
    secret = "4111-1111-1111-1111"
    for line in (
        '{"jsonrpc":"1.0","id":1,"method":"' + secret + '","params":{"ssn":"123-45-6789"}}',
        '{"jsonrpc":"2.0","id":1,"method":"' + secret + '","params":{"ssn":"123-45-6789"}',
        '{"jsonrpc":"2.0","id":1,"method":"' + secret + '\\ud800","params":{"ssn":"123-45-6789"}}',
    ):
        check = check_frame(line)
        assert check.error is not None
        dumped = check.error.model_dump_json()
        assert secret not in dumped
        assert "123-45-6789" not in dumped


async def read_all(pump: LinePump) -> list[bytes]:
    lines = []
    while line := await pump.readline():
        lines.append(line)
    return lines


@pytest.mark.anyio
async def test_line_pump_splits_a_pipe_on_lf_only() -> None:
    big = b'{"reason":"' + b"x" * 200_000 + b'"}\n'  # spans several 64 KiB reads
    chunks = [b'{"a":', b"1}\n", b'{"b":2}\r\n{"c":', b'"3\r4"}\n', big, b"no newline at EOF"]
    read_fd, write_fd = os.pipe()

    def write() -> None:
        with os.fdopen(write_fd, "wb", buffering=0) as writer:
            for chunk in chunks:
                writer.write(chunk)

    writer = threading.Thread(target=write)
    writer.start()
    source = os.fdopen(read_fd, "rb")
    pump = LinePump(source)
    lines = await read_all(pump)
    writer.join(timeout=10)

    # CRLF stays a line ending, a bare CR stays inside its line, the unterminated tail is kept.
    assert lines == [b'{"a":1}\n', b'{"b":2}\r\n', b'{"c":"3\r4"}\n', big, b"no newline at EOF"]
    assert not source.closed
    pump.close()  # the reader reached EOF, so the descriptor is released
    assert source.closed


@pytest.mark.anyio
async def test_line_pump_reads_in_memory_streams() -> None:
    pump = LinePump(io.BytesIO(b"one\ntwo\r\nthree"))
    assert await read_all(pump) == [b"one\n", b"two\r\n", b"three"]


@pytest.mark.anyio
async def test_a_cancelled_read_returns_at_once_and_leaves_its_stream_open() -> None:
    # Ctrl+C while the client keeps stdin open: the pending read is abandoned, not waited for.
    read_fd, write_fd = os.pipe()
    source = os.fdopen(read_fd, "rb")
    pump = LinePump(source)
    with anyio.fail_after(5), anyio.move_on_after(0.2) as wait:
        await pump.readline()  # nothing to read
    assert wait.cancelled_caught
    pump.close()
    assert not source.closed  # the daemon thread may still be blocked reading it
    os.close(write_fd)  # lets that thread see EOF and finish
