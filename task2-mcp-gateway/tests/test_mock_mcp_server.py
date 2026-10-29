"""The mock downstream on its own, through the official SDK client in-process (no HTTP, no gateway)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import mcp_types as types
import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from mcp_gateway.mock_mcp_server import MockToolbox, build_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def toolbox() -> MockToolbox:
    return MockToolbox()


@pytest.fixture
async def client(toolbox: MockToolbox) -> AsyncIterator[Client]:
    async with Client(build_server(toolbox)) as connected:
        yield connected


async def test_tools_are_listed_with_strict_schemas(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == {"get_weather", "admin_reset_key", "admin_delete_user", "whoami"}
    assert tools["admin_reset_key"].input_schema["required"] == ["key_id"]
    assert tools["admin_reset_key"].annotations is not None
    assert tools["admin_reset_key"].annotations.destructive_hint is True


async def test_calls_are_recorded(client: Client, toolbox: MockToolbox) -> None:
    result = await client.call_tool("admin_delete_user", {"user_id": "u-1"})
    assert result.structured_content == {"user_id": "u-1", "status": "deleted"}
    assert [(call.name, call.arguments) for call in toolbox.calls] == [
        ("admin_delete_user", {"user_id": "u-1"})
    ]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("drop_database", {}),
        ("get_weather", {}),
        ("get_weather", {"city": 42}),
        ("admin_reset_key", {"key_id": "k-1", "force": "yes"}),
    ],
)
async def test_bad_calls_are_invalid_params_and_not_recorded(
    client: Client, toolbox: MockToolbox, name: str, arguments: dict[str, Any]
) -> None:
    with pytest.raises(MCPError) as exc:
        await client.call_tool(name, arguments)
    assert exc.value.code == types.INVALID_PARAMS
    assert toolbox.calls == []


async def test_whoami_without_a_gateway_reports_no_identity(client: Client) -> None:
    result = await client.call_tool("whoami", {})
    assert result.structured_content == {"subject": None, "role": None, "authorization_forwarded": False}
