"""Authorization policy: may this caller send this JSON-RPC message? Pure functions, no I/O.

Only `tools/call` is gated: a tool whose name starts with `admin_` requires the `admin` role.
Everything else (`initialize`, `tools/list`, `ping`, notifications, ...) is forwarded unchanged.

The check fails closed, in two layers:

1. `params.name` must follow the MCP tool-name grammar (SEP-986): 1-128 characters from
   `A-Z a-z 0-9 _ - .`. Anything else is `-32602`, for every role. Unicode has too many ways to
   spell `admin_` for a downstream that compares names loosely: with a dotless i (U+0131) it
   upper-cases to `ADMIN_`, and with an accented or combining `a` (U+00E0, U+0300) it becomes
   `admin_` in an ASCII fold. Within the grammar, a name has a single reading.
2. Names are compared after `normalize_identifier`, a case-insensitive comparison for names in
   the grammar. A downstream that matches names exactly treats `Admin_reset_key` as an unknown
   tool, but one that lower-cases would run `admin_reset_key`. The gateway denies the variant to
   viewers either way, without needing to know which kind of downstream it is talking to.

The original name is forwarded untouched.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from mcp_gateway.auth import Role
from mcp_gateway.jsonrpc import HEADER_MISMATCH, INVALID_PARAMS, INVALID_REQUEST, JsonRpcError, Message

TOOLS_CALL = "tools/call"
ADMIN_TOOL_PREFIX = "admin_"
ADMIN_TOOL_ROLES = frozenset({Role.ADMIN})
# SEP-986 tool names. Explicit ASCII ranges: `\w` or `\d` would also match Unicode letters/digits.
TOOL_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}")

# Removed before comparing: control and format characters (zero-width space/joiner, BOM, soft
# hyphen, bidi marks) and every kind of whitespace.
_IGNORED_CATEGORIES = frozenset({"Cc", "Cf", "Zs", "Zl", "Zp"})


def normalize_identifier(value: str) -> str:
    """Fold a method or tool name to a permissive reading a downstream could give it.

    NFKC, case folding, and removal of invisible characters. For a tool name this is the second
    layer: `tool_name` has already limited it to ASCII, where this is plain lower-casing.
    """
    folded = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", value).casefold())
    return "".join(ch for ch in folded if unicodedata.category(ch) not in _IGNORED_CATEGORIES)


def is_admin_tool(name: str) -> bool:
    return normalize_identifier(name).startswith(ADMIN_TOOL_PREFIX)


def is_tool_call(method: str) -> bool:
    return normalize_identifier(method) == TOOLS_CALL


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    tool: str | None = None


def authorize(message: Message, role: Role) -> Decision:
    """Decide whether `role` may send `message`. Invalid tool calls raise `JsonRpcError`."""
    method = message.method
    if method is None:
        return Decision(True, "client_response")
    if not is_tool_call(method):
        return Decision(True, "method_not_gated")
    if message.is_notification:
        # A notification gets no response, so a denial could not even be reported. tools/call is
        # a request by definition: refuse it rather than forward it unchecked.
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request: tools/call must carry an id")
    name = tool_name(message)
    if not is_admin_tool(name):
        return Decision(True, "tool_not_gated", name)
    if role in ADMIN_TOOL_ROLES:
        return Decision(True, "admin_role", name)
    return Decision(False, "admin_tool_requires_admin_role", name)


def tool_name(message: Message) -> str:
    """`params.name` of a tools/call: a single, unambiguous string in the MCP tool-name grammar."""
    params = message.params or {}
    if any(key != "name" and normalize_identifier(key) == "name" for key in params):
        # {"name": "get_weather", "Name": "admin_reset_key"}: a case-insensitive reader could pick
        # the other key.
        raise JsonRpcError(INVALID_PARAMS, "Invalid params: ambiguous tool name key", rpc_id=message.id)
    name = params.get("name")
    if not isinstance(name, str) or not name:
        raise JsonRpcError(
            INVALID_PARAMS, "Invalid params: params.name must be a non-empty string", rpc_id=message.id
        )
    if not TOOL_NAME_PATTERN.fullmatch(name):
        raise JsonRpcError(
            INVALID_PARAMS,
            "Invalid params: params.name must be 1-128 characters from A-Z, a-z, 0-9, '_', '-' and '.'",
            rpc_id=message.id,
            detail=f"tool name outside the MCP grammar: {name!r}"[:200],
        )
    return name


# --- MCP routing headers (protocol 2026-07-28) --------------------------------------------------

MCP_METHOD_HEADER = "mcp-method"
MCP_NAME_HEADER = "mcp-name"
# The body member each method mirrors into Mcp-Name.
NAME_BEARING_PARAMS = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri"}
_BASE64_SENTINEL = re.compile(r"=\?base64\?(?P<payload>[A-Za-z0-9+/]*={0,2})\?=")


def decode_header_value(value: str) -> str | None:
    """Undo the `=?base64?...?=` wrapping MCP uses for non-ASCII header values (None if malformed)."""
    match = _BASE64_SENTINEL.fullmatch(value)
    if match is None:
        return value
    payload = match["payload"]
    try:
        decoded = base64.b64decode(payload, validate=True)
        if base64.b64encode(decoded).decode("ascii") != payload:
            return None  # non-canonical encodings could hide a second meaning
        return decoded.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None


def check_routing_headers(message: Message, headers: Iterable[tuple[str, str]]) -> None:
    """Refuse a request whose MCP routing headers describe a different call than its body.

    MCP 2026-07-28 mirrors the method and tool name into `Mcp-Method` / `Mcp-Name` so that
    intermediaries can route without parsing JSON. The gateway authorizes the body. If a header
    said something else, a component trusting the header could run a call the gateway never
    authorized. Absent headers are fine: earlier protocol versions do not send them.

    `Mcp_Name` counts as `Mcp-Name`: CGI/WSGI servers read both as `HTTP_MCP_NAME`. (The proxy
    also drops every header whose name contains `_` before forwarding.)
    """
    seen: dict[str, str] = {}
    for name, value in headers:
        key = name.lower().replace("_", "-")
        if key in (MCP_METHOD_HEADER, MCP_NAME_HEADER):
            if key in seen:
                raise _mismatch(message, f"{key} header appears more than once")
            seen[key] = value
    if MCP_METHOD_HEADER in seen and seen[MCP_METHOD_HEADER] != message.method:
        raise _mismatch(message, "Mcp-Method header does not match the request body")
    if MCP_NAME_HEADER in seen:
        member = NAME_BEARING_PARAMS.get(message.method or "")
        expected = (message.params or {}).get(member) if member else None
        if expected is None or decode_header_value(seen[MCP_NAME_HEADER]) != expected:
            raise _mismatch(message, "Mcp-Name header does not match the request body")


def _mismatch(message: Message, reason: str) -> JsonRpcError:
    rpc_id = message.id if message.method is not None else None
    return JsonRpcError(HEADER_MISMATCH, reason, rpc_id=rpc_id)
