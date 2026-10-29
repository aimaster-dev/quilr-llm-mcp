"""Audit log: one JSON line per gateway decision, on the `mcp_gateway.audit` logger.

Every request to `/mcp` produces exactly one line:

- `allow`: forwarded to the downstream,
- `deny`: authenticated, but the role may not make this call (-32001),
- `reject`: unauthenticated or malformed, so never evaluated.

Tokens are never logged. Strings that come from the client are truncated and JSON-escaped, so a
hostile tool name can neither forge a second line nor flood the log.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from mcp_gateway.jsonrpc import RequestId

AUDIT_LOGGER_NAME = "mcp_gateway.audit"
MAX_FIELD_LENGTH = 200

logger = logging.getLogger(AUDIT_LOGGER_NAME)

Verdict = Literal["allow", "deny", "reject"]


def _clip(value: str | int | None) -> str | int | None:
    if isinstance(value, str) and len(value) > MAX_FIELD_LENGTH:
        return value[:MAX_FIELD_LENGTH] + "..."
    return value


@dataclass
class AuditRecord:
    """What is known about one request, filled in as the gateway learns it."""

    request_id: str
    http_method: str
    subject: str | None = None
    role: str | None = None
    method: str | None = None
    rpc_id: RequestId | None = None
    tool: str | None = None

    def emit(
        self, decision: Verdict, reason: str, *, status: int | None = None, code: int | None = None
    ) -> None:
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "event": "mcp_gateway.decision",
            "decision": decision,
            "reason": _clip(reason),
            "request_id": self.request_id,
            "subject": self.subject,
            "role": self.role,
            "http_method": self.http_method,
            "method": _clip(self.method),
            "rpc_id": _clip(self.rpc_id),
            "tool": _clip(self.tool),
            "status": status,
            "code": code,
        }
        logger.info(json.dumps(entry, ensure_ascii=True, separators=(",", ":")))
