"""Request schema for ``POST /v1/chat/completions`` (an OpenAI-compatible subset).

Validation is strict (no type coercion) and closed (unknown fields are rejected rather than
silently dropped or forwarded). Only validated fields are sent upstream, so a client cannot smuggle
provider-specific parameters past the gateway.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_router.errors import InvalidRequestError

MAX_TOKENS_LIMIT = 32_768
MAX_MESSAGES = 512
_MAX_ERRORS_SHOWN = 3
_SAFE_LOC_PART = re.compile(r"[^A-Za-z0-9_]")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TextPart(_Strict):
    type: Literal["text"]
    text: str


class ChatMessage(_Strict):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart]
    name: str | None = Field(default=None, min_length=1, max_length=64)


class ChatCompletionRequest(_Strict):
    #: Accepted for SDK compatibility and echoed back; the gateway decides which models serve it.
    model: str | None = Field(default=None, min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_TOKENS_LIMIT)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: str | Annotated[list[str], Field(max_length=4)] | None = None
    stream: bool = False

    def messages_payload(self) -> list[dict[str, Any]]:
        return [message.model_dump(exclude_none=True) for message in self.messages]

    def upstream_payload(self, max_tokens: int) -> dict[str, Any]:
        """Fields forwarded to the provider. ``max_tokens`` is always explicit, so the provider can
        never generate more than the gateway reserved for."""
        payload = self.model_dump(exclude_none=True, exclude={"model", "stream", "max_tokens"})
        payload["messages"] = self.messages_payload()
        payload["max_tokens"] = max_tokens
        return payload


def _location(loc: tuple[int | str, ...]) -> str:
    # Keys can come from the client (e.g. an unknown field name): keep them short and inert.
    parts = [str(p) if isinstance(p, int) else _SAFE_LOC_PART.sub("", p)[:40] or "?" for p in loc]
    return ".".join(parts) or "body"


def parse_chat_request(body: bytes) -> ChatCompletionRequest:
    """Validate a request body, mapping failures to a sanitized ``InvalidRequestError``.

    Error messages name the offending fields and the rule they break, never the submitted values.
    """
    try:
        return ChatCompletionRequest.model_validate_json(body)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False, include_context=False)
        if any(error["type"] == "json_invalid" for error in errors):
            raise InvalidRequestError("The request body is not valid JSON.", code="invalid_json") from None
        if any(error["type"] == "model_type" and not error["loc"] for error in errors):
            raise InvalidRequestError("The request body must be a JSON object.") from None
        shown = "; ".join(f"{_location(e['loc'])}: {e['msg']}" for e in errors[:_MAX_ERRORS_SHOWN])
        more = f" (and {len(errors) - _MAX_ERRORS_SHOWN} more)" if len(errors) > _MAX_ERRORS_SHOWN else ""
        raise InvalidRequestError(f"Invalid request body: {shown}{more}.") from None
