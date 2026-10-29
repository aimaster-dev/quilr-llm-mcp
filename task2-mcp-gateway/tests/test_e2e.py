"""End to end over real HTTP: official MCP SDK client -> gateway -> mock MCP server, both on uvicorn."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from mcp_gateway.auth import Role
from mcp_gateway.config import GatewaySettings
from mcp_gateway.mock_mcp_server import MockToolbox, create_mock_app
from mcp_gateway.proxy import create_app

from .helpers import TEST_SECRET, bearer, serve_in_thread

pytestmark = pytest.mark.anyio

# Two transport flavours, so both response formats and both protocol eras cross the gateway.
STACKS: dict[str, tuple[dict[str, bool], str]] = {
    # MCP 2026-07-28: server/discover, stateless per-request envelopes, JSON responses, and the
    # Mcp-Method / Mcp-Name routing headers.
    "modern-json": ({"sse": False, "stateful": False}, "auto"),
    # Handshake era: initialize, Mcp-Session-Id, SSE responses, a GET stream, DELETE on close.
    "legacy-sse-session": ({"sse": True, "stateful": True}, "legacy"),
}


@dataclass(frozen=True)
class Stack:
    url: str
    toolbox: MockToolbox
    client_mode: str
    sessions: bool

    def calls_by(self, subject: str) -> list[str]:
        return [call.name for call in self.toolbox.calls if call.subject == subject]


@pytest.fixture(scope="module", params=list(STACKS))
def stack(request: pytest.FixtureRequest) -> Iterator[Stack]:
    server_options, client_mode = STACKS[request.param]
    toolbox = MockToolbox()
    with serve_in_thread(create_mock_app(toolbox, **server_options)) as downstream:
        settings = GatewaySettings(
            jwt_secret=TEST_SECRET, downstream_url=f"{downstream}/mcp", connect_timeout=2, read_timeout=10
        )
        with serve_in_thread(create_app(settings)) as gateway:
            yield Stack(f"{gateway}/mcp", toolbox, client_mode, sessions=server_options["stateful"])


@asynccontextmanager
async def connect(stack: Stack, role: Role, subject: str = "alice") -> AsyncIterator[Client]:
    """The official SDK client, sending `Authorization: Bearer <token>` on every request."""
    async with (
        httpx2.AsyncClient(headers=bearer(role, subject), timeout=10) as http,
        Client(streamable_http_client(stack.url, http_client=http), mode=stack.client_mode) as client,
    ):
        yield client


def unique_subject() -> str:
    return f"user-{uuid.uuid4().hex[:8]}"


@pytest.mark.parametrize("role", list(Role))
async def test_tools_list_is_forwarded_transparently(stack: Stack, role: Role) -> None:
    async with connect(stack, role) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == {"get_weather", "admin_reset_key", "admin_delete_user", "whoami"}


@pytest.mark.parametrize("role", list(Role))
async def test_every_role_may_call_ordinary_tools(stack: Stack, role: Role) -> None:
    async with connect(stack, role) as client:
        result = await client.call_tool("get_weather", {"city": "Paris"})
    assert result.is_error is False
    assert result.structured_content == {"city": "Paris", "forecast": "sunny", "temperature_c": 21}


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [("admin_reset_key", {"key_id": "k-1"}), ("admin_delete_user", {"user_id": "u-1"})],
)
async def test_viewer_admin_call_is_refused_and_never_reaches_the_server(
    stack: Stack, tool: str, arguments: dict[str, Any]
) -> None:
    subject = unique_subject()
    async with connect(stack, Role.VIEWER, subject) as client:
        with pytest.raises(MCPError) as exc:
            await client.call_tool(tool, arguments)
        # Only that call failed: the same session keeps working.
        await client.call_tool("get_weather", {"city": "Oslo"})

    assert exc.value.code == -32001
    assert exc.value.message == "Unauthorized Tool Call"
    assert stack.calls_by(subject) == ["get_weather"]


async def test_admin_may_call_admin_tools(stack: Stack) -> None:
    subject = unique_subject()
    async with connect(stack, Role.ADMIN, subject) as client:
        rotated = await client.call_tool("admin_reset_key", {"key_id": "k-42"})
        deleted = await client.call_tool("admin_delete_user", {"user_id": "u-7"})

    assert rotated.structured_content == {"key_id": "k-42", "status": "rotated"}
    assert deleted.structured_content == {"user_id": "u-7", "status": "deleted"}
    assert stack.calls_by(subject) == ["admin_reset_key", "admin_delete_user"]
    assert {call.role for call in stack.toolbox.calls if call.subject == subject} == {"admin"}


async def test_downstream_sees_the_verified_identity_never_the_token(stack: Stack) -> None:
    async with connect(stack, Role.VIEWER, "carol") as client:
        result = await client.call_tool("whoami", {})
    assert result.structured_content == {
        "subject": "carol",
        "role": "viewer",
        "authorization_forwarded": False,
    }


async def test_sse_responses_and_sessions_cross_the_gateway(stack: Stack) -> None:
    """Plain HTTP, as in the README walkthrough: the wire format the downstream chose is kept."""
    initialize = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "curl", "version": "8"},
        },
    }
    headers = {**bearer(Role.VIEWER), "Accept": "application/json, text/event-stream"}
    async with httpx.AsyncClient(trust_env=False, headers=headers, timeout=10) as http:
        initialized = await http.post(stack.url, json=initialize)
        session = initialized.headers.get("mcp-session-id")
        if not stack.sessions:  # the stateless JSON flavour
            assert initialized.headers["content-type"] == "application/json"
            assert initialized.json()["result"]["serverInfo"]["name"] == "mock-downstream"
            assert session is None
            return

        assert initialized.headers["content-type"].startswith("text/event-stream")
        assert initialized.text.splitlines()[0] == "event: message"
        assert session
        in_session = {"Mcp-Session-Id": session, "MCP-Protocol-Version": "2025-11-25"}
        listed = await http.post(
            stack.url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=in_session
        )
        closed = await http.delete(stack.url, headers=in_session)

    assert listed.status_code == 200
    assert '"admin_reset_key"' in listed.text
    assert closed.status_code in (200, 204)


async def test_requests_without_a_token_are_refused(stack: Stack) -> None:
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    async with httpx.AsyncClient(trust_env=False) as http:
        response = await http.post(
            stack.url, json=request, headers={"Accept": "application/json, text/event-stream"}
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="mcp-gateway"'
    assert response.json()["error"]["code"] == -32003
