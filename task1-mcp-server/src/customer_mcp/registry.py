"""Tool registry: binds each MCP tool definition to its input model and handler.

The JSON Schema advertised in `tools/list` is generated from the same Pydantic model that
validates `tools/call` arguments, so the published contract and the enforced contract can
never drift apart.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import mcp_types as types
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

MAX_REPORTED_ERRORS = 20

ToolHandler = Callable[[Any], Awaitable[BaseModel]]


class ToolExecutionError(Exception):
    """A well-formed call that failed for a business reason (reported as `isError: true`).

    Per the MCP spec, these are *tool execution errors*: the request itself was valid, so
    the model should see the failure and may adjust (e.g. pick another customer).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    title: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    annotations: types.ToolAnnotations | None = None

    def definition(self) -> types.Tool:
        return types.Tool(
            name=self.name,
            title=self.title,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(mode="serialization"),
            annotations=self.annotations,
        )


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def definitions(self) -> list[types.Tool]:
        return [spec.definition() for spec in self._specs.values()]

    async def call(self, name: str, arguments: Mapping[str, Any] | None) -> types.CallToolResult:
        spec = self._specs.get(name)
        if spec is None:
            # MCP spec: unknown tools are a protocol error with code -32602.
            raise MCPError(code=types.INVALID_PARAMS, message="Unknown tool", data={"tool": name[:128]})

        try:
            params = spec.input_model.model_validate(dict(arguments or {}))
        except ValidationError as exc:
            # Standard JSON-RPC "Invalid params" with machine-readable, field-level details.
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"Invalid arguments for tool '{name}'",
                data={"tool": name, "errors": describe_validation_errors(exc)},
            ) from None
        try:
            output = await spec.handler(params)
        except ToolExecutionError as exc:
            logger.info("tool %s rejected request: %s", name, exc.code)
            return _error_result(exc.code, exc.message)

        structured = output.model_dump(mode="json")
        return types.CallToolResult(
            content=[types.TextContent(text=json.dumps(structured, separators=(",", ":")))],
            structured_content=structured,
        )


def describe_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    """Summarise a ValidationError without echoing the caller's input values back.

    Inputs can contain sensitive data (and arbitrarily large payloads), so only the field path,
    the error type and Pydantic's generic message are returned.
    """
    details: list[dict[str, str]] = []
    for error in exc.errors(include_url=False, include_input=False, include_context=False):
        field = ".".join(str(part) for part in error["loc"]) or "(root)"
        details.append({"field": field[:64], "type": error["type"], "message": error["msg"]})
        if len(details) >= MAX_REPORTED_ERRORS:
            break
    return details


def _error_result(code: str, message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(text=message)],
        structured_content={"error": {"code": code, "message": message}},
        is_error=True,
    )
