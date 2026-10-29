"""Method-level authorization over HTTP: denials are answered by the gateway and never forwarded."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from mcp_gateway.auth import Role

from .helpers import Downstream, gateway_client

pytestmark = pytest.mark.anyio

DENIED = {"code": -32001, "message": "Unauthorized Tool Call", "data": {"requiredRole": "admin"}}


def tool_call(name: Any, rpc_id: int | str = 7) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": {"key_id": "k-1"}},
    }


async def post_raw(client: httpx.AsyncClient, body: bytes, **headers: str) -> httpx.Response:
    return await client.post("/mcp", content=body, headers={"Content-Type": "application/json", **headers})


def audit_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage()) for record in caplog.records if record.name == "mcp_gateway.audit"
    ]


# --- the rule -----------------------------------------------------------------------------------


@pytest.mark.parametrize("rpc_id", [7, "req-7"])
async def test_viewer_calling_an_admin_tool_is_denied_without_contacting_the_downstream(
    gateway: httpx.AsyncClient, downstream: Downstream, rpc_id: int | str
) -> None:
    response = await gateway.post("/mcp", json=tool_call("admin_reset_key", rpc_id))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"jsonrpc": "2.0", "id": rpc_id, "error": DENIED}
    assert downstream.requests == []


@pytest.mark.parametrize("tool", ["admin_reset_key", "admin_delete_user"])
async def test_admin_may_call_admin_tools(downstream: Downstream, tool: str) -> None:
    async with gateway_client(downstream, role=Role.ADMIN) as gateway:
        response = await gateway.post("/mcp", json=tool_call(tool))

    assert response.json() == {"jsonrpc": "2.0", "id": 7, "result": {"method": "tools/call"}}
    assert json.loads(downstream.last.content) == tool_call(tool)
    assert downstream.last.headers["x-gateway-role"] == "admin"


@pytest.mark.parametrize("role", list(Role))
async def test_ordinary_tools_and_tools_list_are_forwarded_for_every_role(
    downstream: Downstream, role: Role
) -> None:
    tools_list = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"cursor": "c-1"}}
    async with gateway_client(downstream, role=role) as gateway:
        listed = await gateway.post("/mcp", json=tools_list)
        called = await gateway.post("/mcp", json=tool_call("get_weather"))

    assert listed.json()["result"] == {"method": "tools/list"}
    assert called.json()["result"] == {"method": "tools/call"}
    assert [json.loads(request.content) for request in downstream.requests] == [
        tools_list,
        tool_call("get_weather"),
    ]


async def test_notifications_and_client_responses_are_forwarded(
    gateway: httpx.AsyncClient, downstream: Downstream
) -> None:
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    sampling_reply = {"jsonrpc": "2.0", "id": "srv-1", "result": {"content": {"type": "text", "text": "hi"}}}
    for message in (initialized, sampling_reply):
        assert (await gateway.post("/mcp", json=message)).status_code == 202
    assert [json.loads(request.content) for request in downstream.requests] == [initialized, sampling_reply]


# --- bypass attempts ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Admin_reset_key",
        "ADMIN_RESET_KEY",
        " admin_reset_key",
        "admin_reset_key ",
        "\tadmin_delete_user",
        "\u200badmin_reset_key",
        "\ufeffadmin_reset_key",
        "\uff41\uff44\uff4d\uff49\uff4e_reset_key",
        "admin\uff3freset_key",
    ],
)
async def test_admin_name_variants_are_denied_to_viewers(
    gateway: httpx.AsyncClient, downstream: Downstream, name: str
) -> None:
    response = await gateway.post("/mcp", json=tool_call(name))

    assert response.status_code == 200
    assert response.json()["error"] == DENIED
    assert downstream.requests == []


ADMIN_CALL = b'"method":"tools/call","params":{"name":"admin_reset_key"}'


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        pytest.param(
            b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"get_weather","name":"admin_reset_key"}}',
            400,
            -32700,
            id="duplicate-name-key",
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":7,"method":"tools/list","method":"tools/call","params":{"name":"admin_x"}}',
            400,
            -32700,
            id="duplicate-method-key",
        ),
        pytest.param(
            b'[{"jsonrpc":"2.0","id":1,"method":"tools/list"},{"jsonrpc":"2.0","id":2,' + ADMIN_CALL + b"}]",
            400,
            -32600,
            id="batch-hiding-admin-call",
        ),
        pytest.param(b'{"jsonrpc":"2.0",' + ADMIN_CALL + b"}", 400, -32600, id="tools-call-notification"),
        pytest.param(
            b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"\\ud800admin_reset_key"}}',
            400,
            -32700,
            id="lone-surrogate",
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"get_weather","Name":"admin_x"}}',
            400,
            -32602,
            id="case-variant-name-key",
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":["admin_reset_key"]}}',
            400,
            -32602,
            id="array-name",
        ),
        pytest.param(
            b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":null}}',
            400,
            -32602,
            id="null-name",
        ),
        pytest.param(b'{"jsonrpc":"2.0","id":7,"method":"tools/call"}', 400, -32602, id="missing-params"),
        pytest.param(
            b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":["admin_reset_key"]}',
            400,
            -32602,
            id="positional-params",
        ),
        pytest.param(b'{"jsonrpc":"2.0","id":null,' + ADMIN_CALL + b"}", 400, -32600, id="null-id"),
        pytest.param(b'{"jsonrpc":"1.0","id":7,' + ADMIN_CALL + b"}", 400, -32600, id="wrong-version"),
        pytest.param(b"not json", 400, -32700, id="not-json"),
    ],
)
async def test_malformed_payloads_never_reach_the_downstream(
    downstream: Downstream, role: Role, body: bytes, status: int, code: int
) -> None:
    async with gateway_client(downstream, role=role) as gateway:
        response = await post_raw(gateway, body)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert downstream.requests == []


async def test_method_spelling_variants_are_gated(gateway: httpx.AsyncClient, downstream: Downstream) -> None:
    response = await post_raw(
        gateway, b'{"jsonrpc":"2.0","id":7,"method":"Tools/Call","params":{"name":"admin_x"}}'
    )
    assert response.json()["error"] == DENIED
    assert downstream.requests == []


async def test_the_authorized_bytes_are_what_the_downstream_receives(downstream: Downstream) -> None:
    received = (
        b'{ "jsonrpc": "2.0",\n "id": 7, "method": "tools\\u002fcall", '
        b'"params": {"name": "get\\u005fweather", "arguments": {"city": "Z\\u00fcrich"}} }'
    )
    async with gateway_client(downstream, role=Role.VIEWER) as gateway:
        await post_raw(gateway, received)

    forwarded = downstream.last
    assert (
        forwarded.content
        == (
            '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"get_weather","arguments":{"city":"Zürich"}}}'
        ).encode()
    )
    assert forwarded.headers["content-length"] == str(len(forwarded.content))


async def test_routing_header_that_contradicts_the_body_is_rejected(
    gateway: httpx.AsyncClient, downstream: Downstream
) -> None:
    response = await gateway.post(
        "/mcp",
        json=tool_call("get_weather"),
        headers={"Mcp-Method": "tools/call", "Mcp-Name": "admin_reset_key"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32020
    assert downstream.requests == []


async def test_consistent_routing_headers_are_forwarded(
    gateway: httpx.AsyncClient, downstream: Downstream
) -> None:
    headers = {"Mcp-Method": "tools/call", "Mcp-Name": "get_weather", "MCP-Protocol-Version": "2026-07-28"}
    response = await gateway.post("/mcp", json=tool_call("get_weather"), headers=headers)

    assert response.status_code == 200
    assert downstream.last.headers["mcp-name"] == "get_weather"


async def test_downstream_failure_reply_carries_the_request_id(downstream: Downstream) -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    downstream.responder = unreachable
    async with gateway_client(downstream) as gateway:
        response = await gateway.post("/mcp", json=tool_call("get_weather", "req-9"))

    assert response.status_code == 502
    assert response.json()["id"] == "req-9"


# --- audit log ----------------------------------------------------------------------------------


async def test_every_decision_is_audited_once(
    downstream: Downstream, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="mcp_gateway.audit")
    async with gateway_client(downstream, role=Role.VIEWER) as viewer:
        await viewer.post("/mcp", json=tool_call("get_weather", 1))
        await viewer.post("/mcp", json=tool_call("admin_reset_key", 2))
        await post_raw(viewer, b"{oops")
    async with gateway_client(downstream, role=None) as anonymous:
        await anonymous.post("/mcp", json=tool_call("get_weather", 3))

    allow, deny, reject, unauthenticated = audit_lines(caplog)
    common = {"subject": "alice", "role": "viewer", "http_method": "POST"}
    assert (
        allow.items()
        >= {
            "decision": "allow",
            "reason": "tool_not_gated",
            "method": "tools/call",
            "tool": "get_weather",
            "rpc_id": 1,
            **common,
        }.items()
    )
    assert (
        deny.items()
        >= {
            "decision": "deny",
            "reason": "admin_tool_requires_admin_role",
            "method": "tools/call",
            "tool": "admin_reset_key",
            "rpc_id": 2,
            "status": 200,
            "code": -32001,
            **common,
        }.items()
    )
    assert reject.items() >= {"decision": "reject", "code": -32700, "method": None, **common}.items()
    assert (
        unauthenticated.items()
        >= {"decision": "reject", "subject": None, "role": None, "code": -32003}.items()
    )
    assert unauthenticated["reason"].startswith("authentication failed")
    assert len({line["request_id"] for line in (allow, deny, reject, unauthenticated)}) == 4
    assert downstream.last.headers["x-gateway-request-id"] == allow["request_id"]


async def test_audit_lines_never_contain_the_token_and_cannot_be_forged(
    downstream: Downstream, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="mcp_gateway.audit")
    hostile = 'admin_x\n{"decision":"allow"}' + "A" * 1000
    async with gateway_client(downstream, role=Role.VIEWER) as gateway:
        token = gateway.headers["authorization"].removeprefix("Bearer ")
        await gateway.post("/mcp", json=tool_call(hostile))

    records = [record.getMessage() for record in caplog.records if record.name == "mcp_gateway.audit"]
    assert len(records) == 1
    assert "\n" not in records[0]
    line = json.loads(records[0])
    assert line["decision"] == "deny"
    assert len(line["tool"]) < 300
    assert token.split(".")[2] not in records[0]
