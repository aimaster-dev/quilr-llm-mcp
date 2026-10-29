"""The authorization policy, as pure functions: no HTTP, no downstream."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mcp_gateway.auth import Role
from mcp_gateway.jsonrpc import HEADER_MISMATCH, INVALID_PARAMS, INVALID_REQUEST, JsonRpcError, parse_message
from mcp_gateway.policy import (
    authorize,
    check_routing_headers,
    decode_header_value,
    is_admin_tool,
    normalize_identifier,
)


def message(method: str = "tools/call", params: Any = None, **envelope: Any) -> Any:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method, **envelope}
    if params is not None:
        payload["params"] = params
    return parse_message(json.dumps(payload).encode())


def call(name: Any, **params: Any) -> Any:
    return message(params={"name": name, "arguments": {}, **params})


# --- admin tool detection -----------------------------------------------------------------------

ADMIN_VARIANTS = [
    "admin_reset_key",
    "Admin_reset_key",
    "ADMIN_DELETE_USER",
    " admin_reset_key",
    "\tadmin_reset_key\n",
    "\u200badmin_reset_key",  # zero-width space
    "\ufeffadmin_reset_key",  # byte-order mark
    "adm\u00adin_reset_key",  # soft hyphen
    "\u202eadmin_reset_key",  # right-to-left override
    "\uff41\uff44\uff4d\uff49\uff4e_reset_key",  # full-width letters
    "admin\uff3freset_key",  # full-width low line
    "\u00a0admin_x",  # no-break space
    "admin_",
]


@pytest.mark.parametrize("name", ADMIN_VARIANTS)
def test_admin_tool_variants_are_recognised(name: str) -> None:
    assert is_admin_tool(name)


@pytest.mark.parametrize(
    "name", ["get_weather", "administer", "admin", "adminreset", "my_admin_tool", "_admin_x"]
)
def test_ordinary_tools_are_not_admin_tools(name: str) -> None:
    assert not is_admin_tool(name)


def test_normalisation_folds_case_width_and_invisible_characters() -> None:
    assert normalize_identifier(" \uff21\uff24\uff2d\uff29\uff2e\u200b_Reset ") == "admin_reset"


_INVISIBLE = st.sampled_from(
    [" ", "\t", "\n", "\r", "\u00a0", "\u200b", "\u200d", "\u2060", "\ufeff", "\u00ad"]
)


@settings(max_examples=300, deadline=None)
@given(
    prefix=st.lists(_INVISIBLE, max_size=4).map("".join),
    upper=st.lists(st.booleans(), min_size=6, max_size=6),
    suffix=st.text(max_size=20),
)
def test_any_spelling_of_the_admin_prefix_is_an_admin_tool(
    prefix: str, upper: list[bool], suffix: str
) -> None:
    spelled = "".join(ch.upper() if up else ch for ch, up in zip("admin_", upper, strict=True))
    assert is_admin_tool(prefix + spelled + suffix)


@settings(max_examples=300, deadline=None)
@given(st.from_regex(r"[A-Za-z0-9._-]{1,64}", fullmatch=True))
def test_spec_conformant_names_are_admin_tools_iff_they_start_with_admin_(name: str) -> None:
    # SEP-986 tool names: for these the check is a plain case-insensitive prefix test.
    assert is_admin_tool(name) == name.lower().startswith("admin_")


# --- authorize ----------------------------------------------------------------------------------


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize(
    "msg",
    [
        pytest.param(message("tools/list"), id="tools/list"),
        pytest.param(message("initialize", {"protocolVersion": "2025-11-25"}), id="initialize"),
        pytest.param(message("resources/read", {"uri": "admin_://x"}), id="resources/read"),
        pytest.param(message("prompts/get", {"name": "admin_prompt"}), id="prompts/get"),
        pytest.param(call("get_weather"), id="ordinary-tool"),
        pytest.param(
            parse_message(b'{"jsonrpc":"2.0","method":"notifications/initialized"}'), id="notification"
        ),
        pytest.param(parse_message(b'{"jsonrpc":"2.0","id":"s-1","result":{}}'), id="client-response"),
    ],
)
def test_everything_but_admin_tool_calls_is_allowed_for_every_role(msg: Any, role: Role) -> None:
    assert authorize(msg, role).allowed


@pytest.mark.parametrize("name", ["admin_reset_key", "Admin_reset_key", " admin_delete_user"])
def test_admin_tools_are_denied_to_viewers(name: str) -> None:
    decision = authorize(call(name), Role.VIEWER)
    assert not decision.allowed
    assert decision.tool == name
    assert decision.reason == "admin_tool_requires_admin_role"


def test_admin_tools_are_allowed_to_admins() -> None:
    decision = authorize(call("admin_reset_key"), Role.ADMIN)
    assert decision.allowed
    assert decision.tool == "admin_reset_key"


@pytest.mark.parametrize(
    "method",
    ["Tools/Call", "TOOLS/CALL", " tools/call", "tools/call\u200b", "\uff54\uff4f\uff4f\uff4c\uff53/call"],
)
def test_method_spelling_variants_are_still_gated(method: str) -> None:
    assert not authorize(message(method, {"name": "admin_reset_key"}), Role.VIEWER).allowed


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize(
    "msg",
    [
        pytest.param(message(), id="no-params"),
        pytest.param(message(params={}), id="no-name"),
        pytest.param(call(None), id="null-name"),
        pytest.param(call(""), id="empty-name"),
        pytest.param(call(42), id="number-name"),
        pytest.param(call(["admin_reset_key"]), id="array-name"),
        pytest.param(call({"value": "admin_reset_key"}), id="object-name"),
        pytest.param(call("get_weather", Name="admin_reset_key"), id="case-variant-key"),
        pytest.param(call("get_weather", **{" name": "admin_reset_key"}), id="padded-key"),
    ],
)
def test_malformed_tool_calls_are_invalid_params_for_every_role(msg: Any, role: Role) -> None:
    with pytest.raises(JsonRpcError) as exc:
        authorize(msg, role)
    assert exc.value.code == INVALID_PARAMS
    assert exc.value.rpc_id == 1


@pytest.mark.parametrize("role", list(Role))
def test_tool_call_notification_is_an_invalid_request(role: Role) -> None:
    notification = parse_message(
        b'{"jsonrpc":"2.0","method":"tools/call","params":{"name":"admin_reset_key"}}'
    )
    with pytest.raises(JsonRpcError) as exc:
        authorize(notification, role)
    assert exc.value.code == INVALID_REQUEST


# --- MCP routing headers ------------------------------------------------------------------------


def b64(value: str) -> str:
    return f"=?base64?{base64.b64encode(value.encode()).decode()}?="


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [("mcp-method", "tools/call")],
        [("Mcp-Method", "tools/call"), ("Mcp-Name", "get_weather")],
        [("mcp-method", "tools/call"), ("mcp-name", b64("get_weather"))],
    ],
)
def test_consistent_routing_headers_pass(headers: list[tuple[str, str]]) -> None:
    check_routing_headers(call("get_weather"), headers)


@pytest.mark.parametrize(
    "headers",
    [
        [("mcp-method", "tools/list")],
        [("mcp-name", "admin_reset_key")],
        [("mcp-name", b64("admin_reset_key"))],
        [("mcp-name", "=?base64?not base64?=")],
        [("mcp-name", "get_weather"), ("mcp-name", "admin_reset_key")],
        [("mcp-method", "tools/call"), ("Mcp-Method", "tools/call")],
    ],
)
def test_routing_headers_that_contradict_the_body_are_rejected(headers: list[tuple[str, str]]) -> None:
    with pytest.raises(JsonRpcError) as exc:
        check_routing_headers(call("get_weather"), headers)
    assert exc.value.code == HEADER_MISMATCH
    assert exc.value.rpc_id == 1


def test_mcp_name_on_a_method_without_a_name_is_rejected() -> None:
    with pytest.raises(JsonRpcError):
        check_routing_headers(message("tools/list"), [("mcp-name", "admin_reset_key")])


@pytest.mark.parametrize(
    ("raw", "decoded"),
    [
        ("plain", "plain"),
        (b64("zoë"), "zoë"),
        ("=?base64?Zm9v?=", "foo"),
        ("=?base64?Zm9=?=", None),  # non-canonical padding bits
        ("=?base64?/w==?=", None),  # not UTF-8
    ],
)
def test_header_value_decoding(raw: str, decoded: str | None) -> None:
    assert decode_header_value(raw) == decoded
