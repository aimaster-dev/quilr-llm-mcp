"""Tool registry: binds each MCP tool definition to its input model and handler.

The JSON Schema advertised in `tools/list` is generated from the same Pydantic model that
validates `tools/call` arguments, so the published contract and the enforced contract can
never drift apart.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mcp_types as types
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ValidationError

from customer_mcp.schemas import ToolErrorDetail, ToolErrorText

logger = logging.getLogger(__name__)

MAX_REPORTED_ERRORS = 20
# A name is echoed back in error `data` only if it looks like one: MCP's tool-name alphabet for tool
# names, an identifier for argument names. Anything else (control characters, a 100 KB string, an
# argument *named* "SSN 123-45-6789") is replaced, never copied to the wire.
_ECHOABLE_TOOL_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,128}")
_ECHOABLE_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
REDACTED_FIELD = "(redacted)"

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
            echo = {"tool": name} if _ECHOABLE_TOOL_NAME.fullmatch(name) else None
            raise MCPError(code=types.INVALID_PARAMS, message="Unknown tool", data=echo)

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
        except Exception:
            # Unexpected fault: full traceback to stderr, generic -32603 on the wire. Without this
            # the SDK would answer with code 0 and str(exc), leaking internals to the client.
            logger.exception("tool %s failed unexpectedly", name)
            raise MCPError(code=types.INTERNAL_ERROR, message="Internal error") from None

        structured = output.model_dump(mode="json")
        return types.CallToolResult(
            content=[types.TextContent(text=json.dumps(structured, separators=(",", ":")))],
            structured_content=structured,
        )


def describe_validation_errors(
    exc: ValidationError, *, strip_prefix: str | None = None
) -> list[dict[str, str]]:
    """Summarise a ValidationError without echoing the caller's input back.

    Inputs can contain sensitive data (and arbitrarily large payloads), so only the field path,
    the error type and Pydantic's generic message are returned. A path segment that is not a
    plain identifier (an unexpected key the caller chose) is redacted. `strip_prefix` drops a
    leading segment, e.g. the `params` the SDK prepends to request-level paths.
    """
    details: list[dict[str, str]] = []
    for error in exc.errors(include_url=False, include_input=False, include_context=False):
        loc: Sequence[int | str] = error["loc"]
        if strip_prefix is not None and len(loc) > 1 and loc[0] == strip_prefix:
            loc = loc[1:]
        details.append({"field": _field_path(loc), "type": error["type"], "message": error["msg"]})
        if len(details) >= MAX_REPORTED_ERRORS:
            break
    return details


def _field_path(loc: Sequence[int | str]) -> str:
    parts = [
        str(part) if isinstance(part, int) or _ECHOABLE_FIELD_NAME.fullmatch(part) else REDACTED_FIELD
        for part in loc
    ]
    return ".".join(parts) or "(root)"


def _error_result(code: str, message: str) -> types.CallToolResult:
    # No `structuredContent`: the tool's `outputSchema` describes a successful result, and MCP says
    # structured content MUST conform to it. The error is machine-readable JSON in the text instead.
    text = ToolErrorText(error=ToolErrorDetail(code=code, message=message)).model_dump_json()
    return types.CallToolResult(content=[types.TextContent(text=text)], is_error=True)
