"""The authorization policy, as pure functions: no HTTP, no downstream."""

from __future__ import annotations

import base64
import json
import unicodedata
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


# --- admin tool detection (the normalisation layer, on any string) -----------------------------

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
@given(st.from_regex(r"[A-Za-z0-9._-]{1,128}", fullmatch=True))
def test_spec_conformant_names_are_admin_tools_iff_they_start_with_admin_(name: str) -> None:
    # SEP-986 tool names: for these the check is a plain case-insensitive prefix test.
    assert is_admin_tool(name) == name.lower().startswith("admin_")


# --- the MCP tool-name grammar (first layer) ----------------------------------------------------

# Names that a loose downstream could read as an admin tool, but that the NFKC/casefold layer alone
# does not fold to `admin_` (the audit's findings), plus the variants that layer did catch.
UNICODE_LOOKALIKES = [
    "adm\u0131n_reset_key",  # dotless i: "adm\u0131n_reset_key".upper() == "ADMIN_RESET_KEY"
    "adm\u0130n_reset_key",  # capital I with dot above: .lower() is "admi\u0307n_..."
    "\u00e0dmin_reset_key",  # precomposed a with grave: ASCII-folds to "admin_"
    "a\u0300dmin_reset_key",  # a + combining grave
    "admi\u0301n_reset_key",  # combining acute inside the prefix
    "admin\u0301_reset_key",  # combining acute after the prefix
    "\u0430dmin_reset_key",  # Cyrillic a (homoglyph)
    "admin_reset_\u212aey",  # Kelvin sign, lower-cases to "k"
    "admin_re\u017fet_key",  # long s, upper-cases to "S"
    "\uff41\uff44\uff4d\uff49\uff4e_reset_key",  # full-width letters
    "admin\uff3freset_key",  # full-width low line
    "\u200badmin_reset_key",  # zero-width space
    "\ufeffadmin_reset_key",  # byte-order mark
    "adm\u00adin_reset_key",  # soft hyphen
    " admin_reset_key",
    "admin_reset_key ",
    "\tadmin_reset_key\n",
    "\u00a0admin_x",  # no-break space
    "admin_reset_key\x00",
    "get weather",
    "get_weather/../admin_reset_key",
    "x" * 129,
    "admin_" + "x" * 123,
]


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("name", UNICODE_LOOKALIKES)
def test_names_outside_the_tool_name_grammar_are_invalid_params(name: str, role: Role) -> None:
    with pytest.raises(JsonRpcError) as exc:
        authorize(call(name), role)
    assert exc.value.code == INVALID_PARAMS
    assert exc.value.rpc_id == 1
    assert name not in exc.value.message  # the client input is not echoed


@pytest.mark.parametrize(
    ("name", "admin_tool"),
    [
        ("get_weather", False),
        ("get-weather.v2", False),
        ("Z", False),
        ("x" * 128, False),
        ("admin_" + "x" * 122, True),
        ("Admin_reset_key", True),
        ("ADMIN_RESET_KEY", True),
        ("aDmIn_x", True),
        ("admin-reset-key", False),
        ("admin.reset_key", False),
    ],
)
def test_names_in_the_grammar_are_accepted_and_gated_case_insensitively(name: str, admin_tool: bool) -> None:
    assert authorize(call(name), Role.ADMIN).allowed
    viewer = authorize(call(name), Role.VIEWER)
    assert viewer.allowed is not admin_tool
    assert viewer.tool == name


def _ascii_fold(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


_LOOKALIKE_CHARACTERS = ["\u0131", "\u0130", "\u017f", "\u212a", "\u0300", "\u00e0", "\u0430", "\uff41"]
_TRICKY = st.sampled_from([*"adminADMIN_.-x0 \t", *_LOOKALIKE_CHARACTERS, "\u200b", "\u00a0"])


@settings(max_examples=500, deadline=None)
@given(st.one_of(st.text(_TRICKY, min_size=1, max_size=12), st.text(max_size=140)))
def test_every_accepted_tool_name_has_a_single_ascii_reading(name: str) -> None:
    try:
        decision = authorize(call(name), Role.VIEWER)
    except JsonRpcError as exc:
        assert exc.code == INVALID_PARAMS
        return
    # Accepted: the case-insensitive, Unicode-folding and ASCII-folding readings a downstream could
    # give the name all agree, so the gateway's prefix check means the same thing as theirs.
    assert name.isascii()
    readings = {
        name.lower(),
        name.casefold(),
        name.upper().lower(),
        normalize_identifier(name),
        _ascii_fold(name).lower(),
    }
    assert len(readings) == 1
    assert decision.allowed is not readings.pop().startswith("admin_")


@settings(max_examples=300, deadline=None)
@given(
    prefix=st.lists(_INVISIBLE, max_size=4).map("".join),
    upper=st.lists(st.booleans(), min_size=6, max_size=6),
    suffix=st.text(max_size=20),
)
def test_no_spelling_of_the_admin_prefix_is_allowed_for_a_viewer(
    prefix: str, upper: list[bool], suffix: str
) -> None:
    spelled = "".join(ch.upper() if up else ch for ch, up in zip("admin_", upper, strict=True))
    try:
        decision = authorize(call(prefix + spelled + suffix), Role.VIEWER)
    except JsonRpcError as exc:
        assert exc.code == INVALID_PARAMS  # outside the grammar
    else:
        assert not decision.allowed


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


@pytest.mark.parametrize("name", ["admin_reset_key", "Admin_reset_key", "ADMIN_DELETE_USER"])
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
        [("Mcp_Method", "tools/call"), ("Mcp_Name", "get_weather")],
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
        # CGI/WSGI servers read `_` as `-`: these are the same headers to them.
        [("Mcp_Name", "admin_reset_key")],
        [("mcp_method", "tools/list")],
        [("mcp-name", "get_weather"), ("mcp_name", "admin_reset_key")],
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
