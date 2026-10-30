"""Audit log: JSON lines on the `mcp_gateway.audit` logger.

Every request to `/mcp` produces exactly one decision line (`"event": "mcp_gateway.decision"`),
written before anything is forwarded:

- `allow`: forwarded to the downstream,
- `deny`: authenticated, but the role may not make this call (-32001),
- `reject`: unauthenticated, malformed, over a limit, or the client went away, so never forwarded.

A request that was allowed gets exactly one more line (`"event": "mcp_gateway.outcome"`, same
`request_id`) once the downstream has answered (`forwarded`, with the downstream's status) or the
gateway gave up (`downstream_unavailable` 502, `gateway_busy` 503, `downstream_timeout` 504,
`internal_error` 500). For a GET stream that is when the stream opens, not when it ends.

Tokens are never logged. Strings that come from the client are truncated and JSON-escaped, so a
hostile tool name can neither forge a second line nor flood the log.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from mcp_gateway.jsonrpc import RequestId

AUDIT_LOGGER_NAME = "mcp_gateway.audit"
DECISION_EVENT = "mcp_gateway.decision"
OUTCOME_EVENT = "mcp_gateway.outcome"
MAX_FIELD_LENGTH = 200

logger = logging.getLogger(AUDIT_LOGGER_NAME)

Verdict = Literal["allow", "deny", "reject"]
Outcome = Literal[
    "forwarded", "downstream_unavailable", "gateway_busy", "downstream_timeout", "internal_error"
]


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
    decided: bool = field(default=False, init=False)
    """Whether the decision line was written (so a later failure is an outcome, not a decision)."""

    def emit(
        self, decision: Verdict, reason: str, *, status: int | None = None, code: int | None = None
    ) -> None:
        """Write the request's decision line. `status`/`code` describe a reply the gateway made itself."""
        self.decided = True
        self._write({"event": DECISION_EVENT, "decision": decision}, reason, status, code)

    def emit_outcome(
        self, outcome: Outcome, *, status: int, code: int | None = None, reason: str | None = None
    ) -> None:
        """Write the outcome line of an allowed request: the HTTP status the client got."""
        self._write({"event": OUTCOME_EVENT, "outcome": outcome}, reason, status, code)

    def _write(self, head: dict[str, Any], reason: str | None, status: int | None, code: int | None) -> None:
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            **head,
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
