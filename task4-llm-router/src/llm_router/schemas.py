"""Request schema for ``POST /v1/chat/completions`` (an OpenAI-compatible subset).

Validation is strict (no type coercion) and closed (unknown fields are rejected rather than
silently dropped or forwarded). Only validated fields are sent upstream, so a client cannot smuggle
provider-specific parameters past the gateway.

Accepted for SDK compatibility: ``max_completion_tokens`` (the current OpenAI name, charged and
forwarded as ``max_tokens``), ``n`` when it is 1, ``user``, the common sampling fields, and chat
histories with tool calls (assistant ``tool_calls`` with ``content: null``, and ``tool`` results).
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_router.errors import InvalidRequestError

MAX_TOKENS_LIMIT = 32_768
MAX_MESSAGES = 512
_MAX_ERRORS_SHOWN = 3
_SAFE_LOC_PART = re.compile(r"[^A-Za-z0-9_]")
# pydantic names the union member an error came from inside ``loc`` (``content.str``,
# ``content.list[TextPart].0.type``). Those are implementation details, not request fields.
_UNION_TAGS = frozenset({"str", "int", "float", "bool", "list", "dict", "none", "TextPart", "ToolCall"})
_TAG_CHARACTERS = re.compile(r"[\[\](),'\"-]")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TextPart(_Strict):
    type: Literal["text"]
    text: str


class FunctionCall(_Strict):
    name: str = Field(min_length=1, max_length=64)
    arguments: str


class ToolCall(_Strict):
    id: str = Field(min_length=1, max_length=256)
    type: Literal["function"]
    function: FunctionCall


class ChatMessage(_Strict):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    #: Required, except on an assistant message that carries tool_calls (OpenAI sends null there).
    content: str | list[TextPart] | None = None
    name: str | None = Field(default=None, min_length=1, max_length=64)
    tool_calls: list[ToolCall] | None = Field(default=None, min_length=1, max_length=128)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def _check_role_fields(self) -> ChatMessage:
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("only assistant messages can have tool_calls")
        if (self.tool_call_id is not None) != (self.role == "tool"):
            raise ValueError("tool messages need a tool_call_id, and only they can have one")
        if self.content is None and self.tool_calls is None:
            raise ValueError(
                "content is required (it may be null only on an assistant message with tool_calls)"
            )
        return self


class ChatCompletionRequest(_Strict):
    #: Accepted for SDK compatibility and echoed back; the gateway decides which models serve it.
    model: str | None = Field(default=None, min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_TOKENS_LIMIT)
    #: The newer OpenAI name for max_tokens; charged and forwarded as max_tokens.
    max_completion_tokens: int | None = Field(default=None, ge=1, le=MAX_TOKENS_LIMIT)
    #: Accepted only as 1: the gateway returns exactly one choice (checked in parse_chat_request).
    n: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    seed: int | None = Field(default=None, ge=-(2**63), le=2**63 - 1)
    stop: str | Annotated[list[str], Field(max_length=4)] | None = None
    user: str | None = Field(default=None, max_length=256)
    stream: bool = False

    @property
    def completion_budget(self) -> int | None:
        """The client's max_tokens, under either name."""
        return self.max_tokens if self.max_tokens is not None else self.max_completion_tokens

    def messages_payload(self) -> list[dict[str, Any]]:
        return [message.model_dump(exclude_none=True) for message in self.messages]

    def upstream_payload(self, max_tokens: int) -> dict[str, Any]:
        """Fields forwarded to the provider. ``max_tokens`` is always explicit, so the provider can
        never generate more than the gateway reserved for."""
        payload = self.model_dump(
            exclude_none=True,
            exclude={"model", "stream", "max_tokens", "max_completion_tokens", "n"},
        )
        payload["messages"] = self.messages_payload()
        payload["max_tokens"] = max_tokens
        return payload


def _is_union_tag(part: str) -> bool:
    return part in _UNION_TAGS or _TAG_CHARACTERS.search(part) is not None


def _location(loc: tuple[int | str, ...], *, client_key: bool = False) -> str:
    """A dotted path to the offending field, without pydantic's union member tags.

    Keys can come from the client (an unknown field name, flagged with ``client_key``): they are
    kept, but shortened and reduced to inert characters.
    """
    parts: list[str] = []
    for position, part in enumerate(loc):
        if isinstance(part, int):
            parts.append(str(part))
        elif (client_key and position == len(loc) - 1) or not _is_union_tag(part):
            parts.append(_SAFE_LOC_PART.sub("", part)[:40] or "?")
    return ".".join(parts) or "body"


def _describe(error: Any) -> str:
    if error["type"] == "extra_forbidden":
        return f"{_location(error['loc'], client_key=True)}: is not supported by this gateway"
    message = str(error["msg"]).removeprefix("Value error, ")
    return f"{_location(error['loc'])}: {message}"


def parse_chat_request(body: bytes) -> ChatCompletionRequest:
    """Validate a request body, mapping failures to a sanitized ``InvalidRequestError``.

    Error messages name the offending fields and the rule they break, never the submitted values.
    """
    try:
        request = ChatCompletionRequest.model_validate_json(body)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False, include_context=False)
        if any(error["type"] == "json_invalid" for error in errors):
            raise InvalidRequestError("The request body is not valid JSON.", code="invalid_json") from None
        if any(error["type"] == "model_type" and not error["loc"] for error in errors):
            raise InvalidRequestError("The request body must be a JSON object.") from None
        problems = list(dict.fromkeys(_describe(error) for error in errors))  # dedupe, keep order
        shown = "; ".join(problems[:_MAX_ERRORS_SHOWN])
        more = f" (and {len(problems) - _MAX_ERRORS_SHOWN} more)" if len(problems) > _MAX_ERRORS_SHOWN else ""
        raise InvalidRequestError(f"Invalid request body: {shown}{more}.") from None

    if request.n is not None and request.n > 1:
        raise InvalidRequestError("n > 1 is not supported: this gateway returns exactly one choice.")
    if (
        request.max_tokens is not None
        and request.max_completion_tokens is not None
        and request.max_tokens != request.max_completion_tokens
    ):
        raise InvalidRequestError(
            "max_tokens and max_completion_tokens disagree; send one of them (they mean the same here)."
        )
    return request
