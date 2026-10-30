"""A mock downstream MCP server to put behind the gateway (official MCP Python SDK, Streamable HTTP).

Tools:

- `get_weather(city)`: an ordinary tool, any role may call it.
- `admin_reset_key(key_id)`, `admin_delete_user(user_id)`: privileged tools the gateway reserves
  for the `admin` role.
- `whoami()`: echoes the identity headers the gateway asserted, and whether an `Authorization`
  header reached this server (it never should).

Every call is recorded in `MockToolbox.calls`, so tests can prove that a denied call never got
here. The server trusts `X-Gateway-*` headers only because it must be reachable solely through
the gateway; it does no authorization of its own.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mcp_types as types
import uvicorn
from mcp.server import Server, ServerRequestContext
from mcp.shared.exceptions import MCPError
from starlette.applications import Starlette

from mcp_gateway import __version__
from mcp_gateway.logging_config import add_log_level_argument, setup_logging
from mcp_gateway.proxy import ROLE_HEADER, SUBJECT_HEADER

logger = logging.getLogger("mcp_gateway.mock_mcp_server")

SERVER_NAME = "mock-downstream"


def _object_schema(**properties: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": text} for name, text in properties.items()},
        "required": list(properties),
        "additionalProperties": False,
    }


TOOLS = [
    types.Tool(
        name="get_weather",
        description="Current weather for a city.",
        input_schema=_object_schema(city="City name, e.g. Paris"),
        annotations=types.ToolAnnotations(read_only_hint=True),
    ),
    types.Tool(
        name="admin_reset_key",
        description="Rotate an API key. Admin only.",
        input_schema=_object_schema(key_id="Identifier of the key to rotate"),
        annotations=types.ToolAnnotations(destructive_hint=True),
    ),
    types.Tool(
        name="admin_delete_user",
        description="Delete a user account. Admin only.",
        input_schema=_object_schema(user_id="Identifier of the user to delete"),
        annotations=types.ToolAnnotations(destructive_hint=True),
    ),
    types.Tool(
        name="whoami",
        description="The identity the gateway asserted for this call.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        annotations=types.ToolAnnotations(read_only_hint=True),
    ),
]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    subject: str | None
    role: str | None


@dataclass
class MockToolbox:
    """The tool implementations, plus a record of every call that reached the server."""

    calls: list[ToolCall] = field(default_factory=list)

    def call(self, name: str, arguments: dict[str, Any], headers: dict[str, str]) -> types.CallToolResult:
        spec = next((tool for tool in TOOLS if tool.name == name), None)
        if spec is None:
            raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {name}")
        expected = set(spec.input_schema["properties"])
        if set(arguments) != expected or not all(isinstance(value, str) for value in arguments.values()):
            raise MCPError(
                code=types.INVALID_PARAMS, message=f"{name} expects string arguments {sorted(expected)}"
            )

        self.calls.append(ToolCall(name, arguments, headers.get(SUBJECT_HEADER), headers.get(ROLE_HEADER)))
        logger.info("tool call %s by %s (%s)", name, headers.get(SUBJECT_HEADER), headers.get(ROLE_HEADER))
        return _result(self._run(name, arguments, headers))

    @staticmethod
    def _run(name: str, arguments: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        if name == "get_weather":
            return {"city": arguments["city"], "forecast": "sunny", "temperature_c": 21}
        if name == "admin_reset_key":
            return {"key_id": arguments["key_id"], "status": "rotated"}
        if name == "admin_delete_user":
            return {"user_id": arguments["user_id"], "status": "deleted"}
        return {
            "subject": headers.get(SUBJECT_HEADER),
            "role": headers.get(ROLE_HEADER),
            "authorization_forwarded": "authorization" in headers,
        }


def _result(structured: dict[str, Any]) -> types.CallToolResult:
    text = types.TextContent(type="text", text=json.dumps(structured))
    return types.CallToolResult(content=[text], structured_content=structured)


def build_server(toolbox: MockToolbox) -> Server[Any]:
    async def on_list_tools(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=TOOLS)

    async def on_call_tool(
        ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        # On the HTTP transport `ctx.request` is the Starlette request that carried the call.
        headers = dict(getattr(ctx.request, "headers", None) or {})
        return toolbox.call(params.name, params.arguments or {}, headers)

    return Server(SERVER_NAME, version=__version__, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def create_mock_app(
    toolbox: MockToolbox | None = None, *, sse: bool = False, stateful: bool = False, host: str = "127.0.0.1"
) -> Starlette:
    """The mock server as an ASGI app serving `/mcp`.

    The default (stateless, JSON responses) is the easiest to drive with curl. `sse=True` streams
    every response as Server-Sent Events, and `stateful=True` issues `Mcp-Session-Id`s.
    """
    server = build_server(toolbox or MockToolbox())
    return server.streamable_http_app(json_response=not sse, stateless_http=not stateful, host=host)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mock-mcp-server", description="Mock downstream MCP server for the gateway."
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=9000, help="bind port (default: %(default)s)")
    parser.add_argument("--sse", action="store_true", help="answer with SSE streams instead of JSON")
    parser.add_argument("--stateful", action="store_true", help="issue Mcp-Session-Id session ids")
    add_log_level_argument(parser)
    args = parser.parse_args(argv)

    setup_logging(parser, args.log_level)
    logger.info("starting %s on http://%s:%d/mcp", SERVER_NAME, args.host, args.port)
    app = create_mock_app(sse=args.sse, stateful=args.stateful, host=args.host)
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
