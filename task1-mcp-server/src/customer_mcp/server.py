"""MCP server wiring: protocol handlers delegate to the tool registry."""

from __future__ import annotations

from typing import Any

import mcp_types as types
from mcp.server import Server, ServerRequestContext

from customer_mcp import SERVER_NAME, __version__
from customer_mcp.registry import ToolRegistry
from customer_mcp.store import CustomerStore
from customer_mcp.tools import customer_tools

INSTRUCTIONS = (
    "Customer-service tools. Look customers up by ID (format CUST-XXXXX, five digits) before "
    "issuing refunds. Refunds are irreversible: confirm the amount and reason with the user first."
)


def build_server(registry: ToolRegistry) -> Server[Any]:
    async def on_list_tools(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=registry.definitions())

    async def on_call_tool(
        ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        return await registry.call(params.name, params.arguments)

    return Server(
        SERVER_NAME,
        version=__version__,
        title="Customer Service",
        instructions=INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def create_server(store: CustomerStore | None = None) -> Server[Any]:
    """Build the server with the customer tools, backed by the seeded in-memory store by default."""
    return build_server(ToolRegistry(customer_tools(store or CustomerStore.with_seed_data())))
