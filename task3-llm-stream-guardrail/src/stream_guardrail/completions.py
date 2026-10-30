"""PII redaction for OpenAI chat-completion payloads: streamed chunks and complete responses.

Deny by default: every string under a choice's ``delta`` (streaming) or ``message`` is model output
and is redacted, whatever its field name: ``content`` (string or array of content parts),
``refusal``, ``reasoning_content``, ``reasoning``, ``audio.transcript``, tool-call arguments and
fields this module has never heard of. Only structural values are exempt (``_exempt``): ``role``,
``type``, ``id``, ``index`` and the function name of a tool call.

``logprobs`` repeats every generated token, which cannot be redacted token by token, so it is
replaced by ``null``. The gateway also refuses requests that ask for it.

Streaming: each string field is an independent text stream with its own ``StreamRedactor``, keyed
by ``(choice index, field path)``. In a path, a list element is identified by its ``index`` field
when it has one (tool calls), else by its ``type`` (content parts), else by its position. So with
``n > 1`` or parallel tool calls, interleaved streams never share state, and a tool call or a
content part whose deltas arrive at different array positions is still one stream. A text content
part is the same stream as a ``content`` string, since clients read both as the message text.

Other strings of a choice (anything outside ``delta``, such as a legacy ``text`` field) are complete
values, so they are redacted one shot, without hold-back.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from stream_guardrail.redaction import REDACTED, StreamRedactor
from stream_guardrail.sse import SSEEvent

# A field path inside a delta or message: dict keys, and list steps. A list step is
# ("index", k) for an element carrying ``"index": k``, ("type", t) for an element whose type is t,
# or ("at", position) otherwise.
Step = str | tuple[Any, ...]
Path = tuple[Step, ...]
Visit = Callable[[Path, str], str]

# Paths that hold the same text in another shape: ``"content": "..."`` and
# ``"content": [{"type": "text", "text": "..."}]`` are both the message text.
_SAME_STREAM: dict[Path, Path] = {
    ("content", ("type", "text"), "text"): ("content",),
    ("content", ("type", "refusal"), "refusal"): ("refusal",),
}

# String values that describe structure, not text the model wrote. Everything else is redacted.
_STRUCTURAL_KEYS = frozenset({"role", "type", "id", "index"})
_NAMED = frozenset({"function", "function_call", "custom"})  # their "name" is the tool's name
# Choice-level keys that are never model text.
_CHOICE_KEYS = frozenset({"index", "finish_reason"})


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _exempt(key: str, parent: Step | None) -> bool:
    return key in _STRUCTURAL_KEYS or (key == "name" and parent in _NAMED)


def _element_step(item: Any, position: int) -> Step:
    if isinstance(item, dict):
        index = item.get("index")
        if isinstance(index, int) and not isinstance(index, bool):
            return ("index", index)
        kind = item.get("type")
        if isinstance(kind, str):
            return ("type", kind)
    return ("at", position)


def _walk(node: dict[str, Any] | list[Any], path: Path, visit: Visit) -> None:
    """Replace every non-structural, non-empty string inside ``node`` by ``visit(path, text)``."""
    if isinstance(node, dict):
        parent = path[-1] if path else None
        for key, value in node.items():
            if isinstance(value, str):
                if value and not _exempt(key, parent):
                    node[key] = visit((*path, key), value)
            elif isinstance(value, dict | list):
                _walk(value, (*path, key), visit)
        return
    for position, item in enumerate(node):
        step = _element_step(item, position)
        if isinstance(item, str):
            if item:
                node[position] = visit((*path, step), item)
        elif isinstance(item, dict | list):
            _walk(item, (*path, step), visit)


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

    def visit(self, path: Path, text: str) -> str:
        return self(text)

    def deep(self, obj: Any) -> Any:
        """Redact every string value in a JSON document (keys are left alone)."""
        if isinstance(obj, str):
            return self(obj)
        if isinstance(obj, list):
            return [self.deep(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self.deep(v) for k, v in obj.items()}
        return obj

    def fields(self, obj: Any) -> Any:
        """Like ``deep`` for model output: structural values (``role``, ``id``...) are kept."""
        if isinstance(obj, dict | list):
            _walk(obj, (), self.visit)
            return obj
        return self.deep(obj)


def redact_completion(body: Any, replacement: str = REDACTED) -> tuple[Any, int]:
    """Redact a non-streaming chat completion in place. Returns ``(body, redactions)``.

    Payloads that are not a completion (provider-specific shapes, errors) get every string value
    redacted instead.
    """
    redact = _Counter(replacement)
    if not (isinstance(body, dict) and isinstance(body.get("choices"), list)):
        return redact.deep(body), redact.redactions
    choices = body["choices"]
    for position, choice in enumerate(choices):
        if not isinstance(choice, dict):
            choices[position] = redact.deep(choice)
            continue
        for key, value in choice.items():
            if key == "logprobs":
                choice[key] = None
            elif key not in _CHOICE_KEYS:
                choice[key] = redact.fields(value)
    return body, redact.redactions


class ChatStreamRedactor:
    """Rewrites a stream of ``chat.completion.chunk`` events with PII redacted.

    ``process()`` takes one parsed SSE event and returns the events to forward right away, usually
    exactly one. ``finish()`` flushes whatever is still held when the upstream ends without
    ``[DONE]``.

    Text that is still held back for a choice is sent in a synthetic chunk *before* that choice's
    ``finish_reason`` chunk and before ``data: [DONE]``, so clients that stop reading at either
    point still receive every character, whichever field it belongs to.
    """

    def __init__(self, replacement: str = REDACTED) -> None:
        self.replacement = replacement
        # (choice index, stream key) -> redactor; insertion order is the flush order.
        self._streams: dict[tuple[int, Path], StreamRedactor] = {}
        self._paths: dict[tuple[int, Path], Path] = {}  # where the stream's text was last seen
        self._template: dict[str, Any] = {}  # id/object/created/model... for synthetic chunks
        self._closed_redactions = 0

    @property
    def redactions(self) -> int:
        return self._closed_redactions + sum(r.redactions for r in self._streams.values())

    @property
    def buffered(self) -> int:
        """Characters held across all choices and fields (for monitoring and tests)."""
        return sum(r.buffered for r in self._streams.values())

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
        choices: list[dict[str, Any]] = []
        for index in sorted({key[0] for key in self._streams}):
            tails = self._flush_choice(index)
            if tails:  # a choice whose streams hold nothing gets no chunk
                choices += _tail_choices(index, {}, tails)
        return self._synthetic(choices)

    def _process_chunk(self, event: SSEEvent, chunk: dict[str, Any]) -> list[SSEEvent]:
        self._template = {k: v for k, v in chunk.items() if k not in ("choices", "usage")}
        choices: list[Any] = chunk["choices"]
        ahead: list[dict[str, Any]] = []  # choices for synthetic chunks sent before this one
        held_back = False
        one_shot = _Counter(self.replacement)
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                choices[position] = one_shot.deep(choice)
                continue
            index = choice.get("index", position)
            if not isinstance(index, int):
                index = position
            for key, value in choice.items():
                if key == "delta" and isinstance(value, dict):
                    held_back |= self._redact_delta(index, value)
                elif key == "logprobs":
                    choice[key] = None
                elif key not in _CHOICE_KEYS:
                    choice[key] = one_shot.fields(value)
            if choice.get("finish_reason") is None:
                continue
            tails = self._flush_choice(index)
            if not tails:
                continue
            # Move this choice's text (redacted part + flushed tail, in that order) into a chunk
            # sent first. The original chunk keeps finish_reason with an empty delta.
            delta = choice.get("delta")
            ahead += _tail_choices(index, delta if isinstance(delta, dict) else {}, tails)
            choice["delta"] = {}
        self._closed_redactions += one_shot.redactions

        out = self._synthetic(ahead)
        if held_back and _carries_nothing(chunk):
            return out  # every piece of text in it is being held back; nothing to send yet
        event.data = _dumps(chunk)
        out.append(event)
        return out

    def _synthetic(self, choices: list[dict[str, Any]]) -> list[SSEEvent]:
        # A choice index appears at most once per chunk; a second entry goes in a later chunk.
        chunks: list[list[dict[str, Any]]] = []
        for choice in choices:
            for group in chunks:
                if all(c["index"] != choice["index"] for c in group):
                    group.append(choice)
                    break
            else:
                chunks.append([choice])
        return [SSEEvent(data=_dumps({**self._template, "choices": group})) for group in chunks]

    def _redact_delta(self, index: int, delta: dict[str, Any]) -> bool:
        """Redact a delta in place. True if some non-empty text was entirely held back."""
        held_back = False

        def feed(path: Path, text: str) -> str:
            nonlocal held_back
            key = (index, _SAME_STREAM.get(path, path))
            redactor = self._streams.get(key)
            if redactor is None:
                redactor = self._streams[key] = StreamRedactor(self.replacement)
            self._paths[key] = path
            out = redactor.feed(text)
            held_back |= not out
            return out

        _walk(delta, (), feed)
        return held_back

    def _flush_choice(self, index: int) -> list[tuple[Path, str]]:
        tails = []
        for key in [k for k in self._streams if k[0] == index]:
            redactor = self._streams.pop(key)
            path = self._paths.pop(key)
            text = redactor.flush()
            self._closed_redactions += redactor.redactions
            if text:
                tails.append((path, text))
        return tails


def _tail_choices(index: int, delta: dict[str, Any], tails: list[tuple[Path, str]]) -> list[dict[str, Any]]:
    """Append each flushed tail to ``delta`` after the text already in it, as synthetic choices.

    A tail whose path does not fit the delta's shape (a field that was a string in one delta and
    an object in another) goes in a choice of its own, after the first.
    """
    leftovers = [(path, text) for path, text in tails if not _place(delta, path, text)]
    deltas = [delta]
    for path, text in leftovers:
        extra: dict[str, Any] = {}
        _place(extra, path, text)
        deltas.append(extra)
    return [{"index": index, "delta": d, "finish_reason": None} for d in deltas]


def _place(root: dict[str, Any], path: Path, text: str) -> bool:
    """Append ``text`` at ``path`` inside ``root``, creating containers as needed.

    Returns False, without changing ``root``, when an existing value has the wrong shape. Each
    container it creates matches the next step, so placing into an empty dict always succeeds.
    """
    node: Any = root
    for depth, step in enumerate(path):
        last = depth == len(path) - 1
        if isinstance(step, str):
            if not isinstance(node, dict):
                return False
            current = node.get(step)
            if last:
                if current is not None and not isinstance(current, str):
                    return False
                node[step] = (current or "") + text
                return True
            if current is None or current == "":
                current = node[step] = _container(path[depth + 1])
            node = current
            continue
        if not isinstance(node, list):
            return False
        kind, ident = step
        if kind == "at" and last:  # a string element of a list
            if ident < len(node) and isinstance(node[ident], str):
                node[ident] += text
            else:
                node.append(text)
            return True
        # Only dicts carry an index or a type, so those steps are never last. The text goes after
        # the existing text: into the last matching element.
        found = None
        for position in reversed(range(len(node))):
            if _element_step(node[position], position) == step:
                found = node[position]
                break
        if found is None:
            found = _container(path[depth + 1])
            if kind != "at" and isinstance(found, dict):
                found[kind] = ident  # tool calls need their index, content parts their type
            node.append(found)
        node = found
    return False


def _container(next_step: Step) -> Any:
    return {} if isinstance(next_step, str) else []


def _carries_nothing(chunk: dict[str, Any]) -> bool:
    """True if the chunk has no usage, no finish_reason or other choice data, only empty text."""
    if chunk.get("usage"):
        return False
    for choice in chunk["choices"]:
        if not isinstance(choice, dict):
            return False
        for key, value in choice.items():
            if key == "delta":
                if not _empty(value):
                    return False
            elif key != "index" and value is not None:
                return False
    return True


def _empty(value: Any) -> bool:
    """True for a delta value made only of empty text, nulls and list-element identities."""
    if value is None or value == "":
        return True
    if isinstance(value, dict):
        return all(key in ("index", "type") or _empty(v) for key, v in value.items())
    if isinstance(value, list):
        return all(_empty(v) for v in value)
    return False
