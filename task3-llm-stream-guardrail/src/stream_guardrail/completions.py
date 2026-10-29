"""PII redaction for OpenAI chat-completion payloads: streamed chunks and complete responses.

Covered text: ``content`` and ``refusal`` of each choice, and ``tool_calls[].function.arguments``.
Each of these is an independent text stream with its own ``StreamRedactor``. They are keyed by
``choices[].index`` (and the tool-call index), so with ``n > 1`` interleaved choices never share
state.
"""

from __future__ import annotations

import json
from typing import Any

from stream_guardrail.redaction import REDACTED, StreamRedactor
from stream_guardrail.sse import SSEEvent

TEXT_FIELDS = ("content", "refusal")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class _Counter:
    """One-shot redaction that also counts what it replaced."""

    def __init__(self, replacement: str) -> None:
        self.replacement = replacement
        self.redactions = 0

    def __call__(self, text: str) -> str:
        redactor = StreamRedactor(self.replacement)
        out = redactor.feed(text) + redactor.flush()
        self.redactions += redactor.redactions
        return out

    def deep(self, obj: Any) -> Any:
        """Redact every string value in a JSON document (keys are left alone)."""
        if isinstance(obj, str):
            return self(obj)
        if isinstance(obj, list):
            return [self.deep(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self.deep(v) for k, v in obj.items()}
        return obj


def redact_completion(body: Any, replacement: str = REDACTED) -> tuple[Any, int]:
    """Redact a non-streaming chat completion in place. Returns ``(body, redactions)``.

    Payloads that are not a completion (provider-specific shapes, errors) get every string value
    redacted instead.
    """
    redact = _Counter(replacement)
    if not (isinstance(body, dict) and isinstance(body.get("choices"), list)):
        return redact.deep(body), redact.redactions
    for choice in body["choices"]:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            for name in TEXT_FIELDS:
                if isinstance(message.get(name), str):
                    message[name] = redact(message[name])
            for call in _list(message.get("tool_calls")):
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                    function["arguments"] = redact(function["arguments"])
    return body, redact.redactions


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


class ChatStreamRedactor:
    """Rewrites a stream of ``chat.completion.chunk`` events with PII redacted.

    ``process()`` takes one parsed SSE event and returns the events to forward right away, usually
    exactly one. ``finish()`` flushes whatever is still held when the upstream ends without
    ``[DONE]``.

    Text that is still held back for a choice is sent in a synthetic chunk *before* that choice's
    ``finish_reason`` chunk and before ``data: [DONE]``, so clients that stop reading content at
    either point still receive every character.
    """

    def __init__(self, replacement: str = REDACTED) -> None:
        self.replacement = replacement
        self._text: dict[tuple[int, str], StreamRedactor] = {}
        self._tool_args: dict[tuple[int, int], StreamRedactor] = {}
        self._template: dict[str, Any] = {}  # id/object/created/model... for synthetic chunks
        self._closed_redactions = 0

    @property
    def redactions(self) -> int:
        live = sum(r.redactions for r in self._text.values()) + sum(
            r.redactions for r in self._tool_args.values()
        )
        return self._closed_redactions + live

    @property
    def buffered(self) -> int:
        """Characters held across all choices (for monitoring and tests)."""
        return sum(r.buffered for r in self._text.values()) + sum(
            r.buffered for r in self._tool_args.values()
        )

    def process(self, event: SSEEvent) -> list[SSEEvent]:
        if event.data is None:
            return [event]  # comments, keep-alives and event-only blocks carry no model text
        if event.data.strip() == "[DONE]":
            return [*self.finish(), event]
        try:
            chunk = json.loads(event.data)
        except ValueError:
            redact = _Counter(self.replacement)
            event.data = redact(event.data)  # not JSON: treat as opaque text
            self._closed_redactions += redact.redactions
            return [event]
        if not (isinstance(chunk, dict) and isinstance(chunk.get("choices"), list)):
            redact = _Counter(self.replacement)
            event.data = _dumps(redact.deep(chunk))  # e.g. an {"error": ...} event
            self._closed_redactions += redact.redactions
            return [event]
        return self._process_chunk(event, chunk)

    def finish(self) -> list[SSEEvent]:
        """Flush every choice that still holds text, as one synthetic chunk."""
        indexes = sorted({key[0] for key in self._text} | {key[0] for key in self._tool_args})
        choices = []
        for index in indexes:
            tail = self._flush_choice(index)
            if tail:
                choices.append({"index": index, "delta": tail, "finish_reason": None})
        if not choices:
            return []
        return [SSEEvent(data=_dumps({**self._template, "choices": choices}))]

    def _process_chunk(self, event: SSEEvent, chunk: dict[str, Any]) -> list[SSEEvent]:
        self._template = {k: v for k, v in chunk.items() if k not in ("choices", "usage")}
        choices: list[Any] = chunk["choices"]
        ahead: list[dict[str, Any]] = []  # choices for a synthetic chunk sent before this one
        held_back = False
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", position)
            if not isinstance(index, int):
                index = position
            delta = choice.get("delta")
            if isinstance(delta, dict):
                held_back |= self._redact_delta(index, delta)
            if choice.get("finish_reason") is None:
                continue
            tail = self._flush_choice(index)
            if not tail:
                continue
            # Move this choice's text (redacted part + flushed tail, in that order) into a chunk
            # sent first. The original chunk keeps finish_reason with an empty delta.
            delta = delta if isinstance(delta, dict) else {}
            _merge_tail(delta, tail)
            moved = {"index": index, "delta": delta}
            if "logprobs" in choice:
                moved["logprobs"] = choice["logprobs"]
            moved["finish_reason"] = None
            ahead.append(moved)
            rest = {k: v for k, v in choice.items() if k not in ("delta", "logprobs")}
            rest["delta"] = {}
            choices[position] = rest

        out = []
        if ahead:
            out.append(SSEEvent(data=_dumps({**self._template, "choices": ahead})))
        if held_back and _carries_nothing(chunk):
            return out  # every piece of text in it is being held back; nothing to send yet
        event.data = _dumps(chunk)
        out.append(event)
        return out

    def _redactor(self, table: dict[Any, StreamRedactor], key: Any) -> StreamRedactor:
        redactor = table.get(key)
        if redactor is None:
            redactor = table[key] = StreamRedactor(self.replacement)
        return redactor

    def _redact_delta(self, index: int, delta: dict[str, Any]) -> bool:
        """Redact a delta in place. True if some non-empty text was entirely held back."""
        held_back = False
        for name in TEXT_FIELDS:
            text = delta.get(name)
            if isinstance(text, str) and text:
                delta[name] = self._redactor(self._text, (index, name)).feed(text)
                held_back |= not delta[name]
        for position, call in enumerate(_list(delta.get("tool_calls"))):
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                tool_index = call.get("index", position)
                key = (index, tool_index if isinstance(tool_index, int) else position)
                function["arguments"] = self._redactor(self._tool_args, key).feed(arguments)
                held_back |= not function["arguments"]
        return held_back

    def _flush_choice(self, index: int) -> dict[str, Any]:
        tail: dict[str, Any] = {}
        for name in TEXT_FIELDS:
            redactor = self._text.pop((index, name), None)
            if redactor is not None:
                text = redactor.flush()
                self._closed_redactions += redactor.redactions
                if text:
                    tail[name] = text
        calls = []
        for key in sorted(k for k in self._tool_args if k[0] == index):
            redactor = self._tool_args.pop(key)
            text = redactor.flush()
            self._closed_redactions += redactor.redactions
            if text:
                calls.append({"index": key[1], "function": {"arguments": text}})
        if calls:
            tail["tool_calls"] = calls
        return tail


def _merge_tail(delta: dict[str, Any], tail: dict[str, Any]) -> None:
    for name in TEXT_FIELDS:
        if name in tail:
            delta[name] = (delta.get(name) or "") + tail[name]
    if "tool_calls" not in tail:
        return
    calls = delta.get("tool_calls")
    if not isinstance(calls, list):
        delta["tool_calls"] = tail["tool_calls"]
        return
    by_index = {c.get("index", p): c for p, c in enumerate(calls) if isinstance(c, dict)}
    for flushed in tail["tool_calls"]:
        current = by_index.get(flushed["index"])
        function = current.get("function") if current is not None else None
        if isinstance(function, dict):
            function["arguments"] = (function.get("arguments") or "") + flushed["function"]["arguments"]
        else:
            calls.append(flushed)


def _carries_nothing(chunk: dict[str, Any]) -> bool:
    """True if the chunk has no usage, no finish_reason, no logprobs and only empty text."""
    if chunk.get("usage"):
        return False
    for choice in chunk["choices"]:
        if not isinstance(choice, dict):
            return False
        if choice.get("finish_reason") is not None or choice.get("logprobs") is not None:
            return False
        delta = choice.get("delta")
        if delta is None:
            continue
        if not isinstance(delta, dict):
            return False
        for key, value in delta.items():
            if key == "tool_calls":
                if not all(_empty_tool_call(call) for call in _list(value)):
                    return False
            elif key in TEXT_FIELDS:
                if value:
                    return False
            elif value is not None:
                return False  # role, audio, ...
    return True


def _empty_tool_call(call: Any) -> bool:
    if not isinstance(call, dict):
        return False
    for key, value in call.items():
        if key == "index":
            continue
        if key == "function" and isinstance(value, dict) and set(value) <= {"arguments"}:
            if value.get("arguments"):
                return False
            continue
        return False
    return True
