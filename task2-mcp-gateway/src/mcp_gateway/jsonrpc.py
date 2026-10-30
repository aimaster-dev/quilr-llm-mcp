"""JSON-RPC 2.0 wire format: strict parsing, canonical serialisation and error objects.

The gateway authorizes a request on its *parsed* form, so the downstream must see exactly that
form. Two JSON parsers can read the same bytes differently (duplicate keys, `NaN`, lone
surrogates, non-UTF-8 encodings, ...), so `parse_message` refuses every input where they might,
and the gateway forwards `Message.canonical` (re-serialised from the parsed object) rather than
the bytes it received.

Numbers are the one thing re-serialisation would otherwise change: a float round trip turns
`12345678901234567890.5` into `1.2345678901234567e+19` and `1E2` into `100.0`. A non-integer
number therefore keeps the text it was written as (`JsonNumber.lexeme`), and that text is what
is forwarded. It is safe to copy: the parser only accepts the JSON number grammar (ASCII digits),
the value is checked to be finite, and nothing the gateway authorizes depends on a number.
Integers are exact already (Python ints have arbitrary precision); only `-0` becomes `0`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RequestId = str | int

# Standard JSON-RPC 2.0 codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Gateway codes, from the implementation-defined server-error band (-32000..-32019).
UNAUTHORIZED_TOOL_CALL = -32001
AUTHENTICATION_FAILED = -32003
DOWNSTREAM_UNAVAILABLE = -32004
DOWNSTREAM_TIMEOUT = -32005
GATEWAY_BUSY = -32006  # every pooled downstream connection is in use (HTTP 503)
TOO_MANY_STREAMS = -32007  # the caller already holds its maximum of open GET streams (HTTP 429)

# MCP 2026-07-28: the HTTP routing headers disagree with the body.
HEADER_MISMATCH = -32020

_REQUEST_MEMBERS = frozenset({"jsonrpc", "id", "method", "params"})
_RESPONSE_MEMBERS = frozenset({"jsonrpc", "id", "result", "error"})


class JsonRpcError(Exception):
    """An error the gateway answers itself, as a JSON-RPC error object.

    `http_status` and `headers` describe the HTTP response that carries it. `rpc_id` is the id of
    the request being answered, or `None` when it is unknown (the body was never parsed or had no
    usable id); JSON-RPC then requires `"id": null`. `detail` is extra context for the gateway's
    own log and is never sent to the client.
    """

    def __init__(
        self,
        code: int,
        message: str,
        *,
        http_status: int = 400,
        rpc_id: RequestId | None = None,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.rpc_id = rpc_id
        self.data = data
        self.headers = dict(headers or {})
        self.detail = detail

    def payload(self) -> dict[str, Any]:
        return error_payload(self.rpc_id, self.code, self.message, self.data)


def error_payload(rpc_id: RequestId | None, code: int, message: str, data: Any = None) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response object."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": error}


@dataclass(frozen=True)
class Message:
    """One validated JSON-RPC 2.0 message: a request, a notification or a response."""

    payload: dict[str, Any]
    canonical: bytes
    """The bytes the gateway forwards: `payload` re-serialised as compact UTF-8 JSON (`canonical_json`)."""

    @property
    def method(self) -> str | None:
        return self.payload.get("method")

    @property
    def id(self) -> RequestId | None:
        return self.payload.get("id")

    @property
    def params(self) -> dict[str, Any] | None:
        return self.payload.get("params")

    @property
    def is_notification(self) -> bool:
        return self.method is not None and "id" not in self.payload


def is_request_id(value: Any) -> bool:
    """MCP request ids are strings or integers: not null, not floats, not booleans."""
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


class _DuplicateKeyError(ValueError):
    pass


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # Parsers disagree on `{"name": "a", "name": "b"}` (first wins, last wins, error), so a
    # downstream could act on a value the gateway never saw. Refuse the object outright.
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise _DuplicateKeyError(key)
        obj[key] = value
    return obj


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


class JsonNumber(float):
    """A non-integer JSON number: its value as a float, and the exact text it was written as."""

    __slots__ = ("lexeme",)
    lexeme: str


# RFC 8259 section 6, with ASCII digits only (`\d` would also match other scripts' digits).
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")


def _json_number(text: str) -> JsonNumber:
    # The JSON scanner only ever passes a number lexeme; it is checked again because it is
    # forwarded verbatim.
    if not _NUMBER.fullmatch(text):
        raise ValueError("malformed number")
    number = JsonNumber(text)
    if not math.isfinite(number):  # 1e999 parses to inf, which has no JSON representation
        raise ValueError("number out of range")
    number.lexeme = text
    return number


class _Text(str):
    """Output that `canonical_json` emits as is (punctuation, encoded keys), not as a JSON string."""


_LBRACE, _RBRACE, _LBRACKET, _RBRACKET, _COMMA = map(_Text, "{}[],")
_json_string = json.JSONEncoder(ensure_ascii=False).encode  # str -> its JSON string literal


def canonical_json(value: Any) -> str:
    """Compact JSON for a parsed message (dicts, lists, str, int, bool, None, `JsonNumber`).

    The same text as `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`, except that a
    `JsonNumber` is written as its lexeme. Iterative, so it handles any depth the parser accepted.
    """
    out: list[str] = []
    pending: list[Any] = [value]  # a stack, popped from the end: pushed in reverse order
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind is _Text:
            out.append(item)
        elif kind is str:
            out.append(_json_string(item))
        elif kind is dict:
            pending.append(_RBRACE)
            for index, (key, member) in enumerate(reversed(item.items())):
                if index:
                    pending.append(_COMMA)
                pending.append(member)
                pending.append(_Text(_json_string(key) + ":"))
            pending.append(_LBRACE)
        elif kind is list:
            pending.append(_RBRACKET)
            for index, member in enumerate(reversed(item)):
                if index:
                    pending.append(_COMMA)
                pending.append(member)
            pending.append(_LBRACKET)
        elif item is None:
            out.append("null")
        elif kind is bool:
            out.append("true" if item else "false")
        elif kind is int:
            out.append(int.__repr__(item))
        elif kind is JsonNumber:
            out.append(item.lexeme)
        else:
            raise TypeError(f"not a parsed JSON value: {kind.__name__}")
    return "".join(out)


def parse_message(body: bytes) -> Message:
    """Parse and validate one JSON-RPC message, or raise `JsonRpcError` (-32700/-32600/-32602)."""
    try:
        # Decoded explicitly: json.loads(bytes) would also accept UTF-16/32, which the downstream
        # may not. A UTF-8 byte-order mark is then rejected by the JSON parser.
        payload = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_reject_constant,
            parse_float=_json_number,
        )
        # Strict encoding rejects lone surrogates ("\ud800"), which other parsers replace or drop.
        canonical_bytes = canonical_json(payload).encode("utf-8")
    except _DuplicateKeyError as exc:
        detail = f"duplicate object key {exc.args[0]!r}"[:200]
        raise JsonRpcError(PARSE_ERROR, "Parse error: duplicate object key", detail=detail) from exc
    except (ValueError, RecursionError) as exc:  # includes JSONDecodeError and UnicodeError
        detail = f"{type(exc).__name__}: {exc}"[:200]
        raise JsonRpcError(PARSE_ERROR, "Parse error", detail=detail) from exc

    if isinstance(payload, list):
        # MCP 2025-06-18 removed JSON-RPC batching. Accepting it would mean authorizing every
        # element; refusing it means no admin call can hide inside a batch.
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request: batch requests are not supported")
    if not isinstance(payload, dict):
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request: expected a JSON object")
    _validate_envelope(payload)
    return Message(payload=payload, canonical=canonical_bytes)


def _validate_envelope(payload: dict[str, Any]) -> None:
    rpc_id = payload.get("id")
    usable_id = rpc_id if is_request_id(rpc_id) else None

    def invalid(reason: str) -> JsonRpcError:
        return JsonRpcError(INVALID_REQUEST, f"Invalid Request: {reason}", rpc_id=usable_id)

    if payload.get("jsonrpc") != "2.0":
        raise invalid('"jsonrpc" must be "2.0"')

    if "method" in payload:  # request or notification
        if payload.keys() - _REQUEST_MEMBERS:
            raise invalid("unexpected top-level member")
        if not isinstance(payload["method"], str) or not payload["method"]:
            raise invalid('"method" must be a non-empty string')
        if "id" in payload and not is_request_id(rpc_id):
            raise invalid('"id" must be a string or an integer')
        params = payload.get("params", {})
        if isinstance(params, list):
            raise JsonRpcError(
                INVALID_PARAMS, "Invalid params: MCP params must be an object", rpc_id=usable_id
            )
        if not isinstance(params, dict):
            raise invalid('"params" must be an object')
        return

    # A response, sent by the client to answer a server-initiated request (sampling, elicitation).
    if payload.keys() - _RESPONSE_MEMBERS or ("result" in payload) == ("error" in payload):
        raise invalid("expected a request, a notification or a response")
    # JSON-RPC allows `"id": null` only on an error response to a request whose id was unreadable.
    if not (is_request_id(rpc_id) or ("id" in payload and rpc_id is None and "error" in payload)):
        raise invalid('"id" must be a string or an integer')
