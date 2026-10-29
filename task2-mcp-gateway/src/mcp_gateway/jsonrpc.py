"""JSON-RPC 2.0 wire format: error codes and error objects produced by the gateway."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RequestId = str | int

# Standard JSON-RPC 2.0 codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Gateway codes, from the implementation-defined server-error band (-32000..-32019).
DOWNSTREAM_UNAVAILABLE = -32004
DOWNSTREAM_TIMEOUT = -32005


class JsonRpcError(Exception):
    """An error the gateway answers itself, as a JSON-RPC error object.

    `http_status` and `headers` describe the HTTP response that carries it. `rpc_id` is the id of
    the request being answered, or `None` when it is unknown (the body was never parsed or had no
    usable id); JSON-RPC then requires `"id": null`.
    """

    def __init__(
        self,
        code: int,
        message: str,
        *,
        http_status: int = 400,
        rpc_id: RequestId | None = None,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.rpc_id = rpc_id
        self.data = data
        self.headers = dict(headers or {})

    def payload(self) -> dict[str, Any]:
        return error_payload(self.rpc_id, self.code, self.message, self.data)


def error_payload(rpc_id: RequestId | None, code: int, message: str, data: Any = None) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response object."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": error}
