"""MCP server wiring: protocol handlers delegate to the tool registry."""

from __future__ import annotations

import re
from typing import Any

import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.context import CallNext, HandlerResult
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from customer_mcp import SERVER_NAME, __version__
from customer_mcp.registry import ToolRegistry, describe_validation_errors
from customer_mcp.store import CustomerStore
from customer_mcp.tools import customer_tools

INSTRUCTIONS = (
    "Customer-service tools. Look customers up by ID (format CUST-XXXXX, five digits) before "
    "issuing refunds. Refunds are irreversible: confirm the amount and reason with the user first."
)

# Method names are echoed in -32601 only when they look like one (MCP methods are short paths).
_ECHOABLE_METHOD = re.compile(r"[A-Za-z0-9_./\-]{1,128}")


async def protocol_error_details(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
    """Server middleware that tightens two errors the SDK produces before any handler runs.

    * `-32601 Method not found`: the SDK puts the whole method name in `data`, unbounded and
      unfiltered (a 100 KB name came back verbatim). It is echoed only when it looks like a method.
    * `-32602` for params that fail the SDK's own validation (`arguments` that is not an object, a
      non-string tool `name`): the SDK sends an empty `data`. This adds the same field-level details
      as tool argument errors, without echoing values.
    """
    try:
        return await call_next(ctx)
    except MCPError as exc:
        if exc.code != types.METHOD_NOT_FOUND:
            raise
        echo = ctx.method if _ECHOABLE_METHOD.fullmatch(ctx.method) else None
        raise MCPError(code=types.METHOD_NOT_FOUND, message="Method not found", data=echo) from None
    except ValidationError as exc:
        if ctx.request_id is None:
            raise  # notifications are never answered
        raise MCPError(
            code=types.INVALID_PARAMS,
            message="Invalid request parameters",
            data={"errors": describe_validation_errors(exc, strip_prefix="params")},
        ) from None


def build_server(registry: ToolRegistry) -> Server[Any]:
    async def on_list_tools(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=registry.definitions())

    async def on_call_tool(
        ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        return await registry.call(params.name, params.arguments)

    server: Server[Any] = Server(
        SERVER_NAME,
        version=__version__,
        title="Customer Service",
        instructions=INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    # Innermost, after the SDK's default OpenTelemetry middleware, which then records the final error.
    server.middleware.append(protocol_error_details)
    return server


def create_server(store: CustomerStore | None = None) -> Server[Any]:
    """Build the server with the customer tools, backed by the seeded in-memory store by default."""
    return build_server(ToolRegistry(customer_tools(store or CustomerStore.with_seed_data())))
