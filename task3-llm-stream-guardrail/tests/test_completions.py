"""Redaction of OpenAI chat-completion chunks and responses."""

from __future__ import annotations

import json
from typing import Any

from stream_guardrail.completions import ChatStreamRedactor, redact_completion
from stream_guardrail.redaction import REDACTED
from stream_guardrail.sse import SSEEvent

R = REDACTED
BASE = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "m"}


def chunk(*choices: dict[str, Any], **extra: Any) -> SSEEvent:
    return SSEEvent(data=json.dumps({**BASE, "choices": list(choices), **extra}))


def delta(index: int = 0, finish: str | None = None, **fields: Any) -> dict[str, Any]:
    return {"index": index, "delta": fields, "finish_reason": finish}


def run(events: list[SSEEvent]) -> list[Any]:
    """Process events and decode the output: JSON payloads as dicts, other data as strings."""
    redactor = ChatStreamRedactor()
    out = [e for event in events for e in redactor.process(event)] + redactor.finish()
    decoded = []
    for event in out:
        try:
            decoded.append(json.loads(event.data))
        except ValueError:
            decoded.append(event.data)
    return decoded


def text_of(outputs: list[Any], index: int = 0, field: str = "content") -> str:
    return "".join(
        choice["delta"].get(field) or ""
        for out in outputs
        if isinstance(out, dict)
        for choice in out.get("choices", [])
        if choice["index"] == index
    )


def test_pii_split_across_deltas_is_redacted() -> None:
    pieces = ["Write to john", ".doe@exa", "mple.com", " or call", " 123-45-", "6789", " now."]
    outputs = run([chunk(delta(role="assistant", content=""))] + [chunk(delta(content=p)) for p in pieces])
    assert text_of(outputs) == f"Write to {R} or call {R} now."
    assert outputs[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}


def test_chunks_whose_text_is_all_held_back_are_dropped() -> None:
    outputs = run([chunk(delta(content="Hi ")), chunk(delta(content="john")), chunk(delta(content="@x.io."))])
    # "john" alone is held back (could be an email), so its chunk carries nothing and is dropped.
    assert len(outputs) == 2
    assert text_of(outputs) == f"Hi {R}."


def test_finish_reason_chunk_is_preceded_by_the_flushed_tail() -> None:
    outputs = run(
        [
            chunk(delta(content="mail me at jane@example.org")),
            chunk(delta(finish="stop"), usage=None),
            SSEEvent(data="[DONE]"),
        ]
    )
    *body, synthetic, finish, done = outputs
    assert text_of(body) == "mail me at "
    assert synthetic["choices"] == [{"index": 0, "delta": {"content": R}, "finish_reason": None}]
    assert synthetic["id"] == BASE["id"]
    assert finish["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert done == "[DONE]"


def test_text_in_the_finish_chunk_keeps_its_order() -> None:
    outputs = run([chunk(delta(content="call 555 ")), chunk(delta(content="0100 x@y.io", finish="stop"))])
    assert text_of(outputs) == f"call 555 0100 {R}"
    assert outputs[-1]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}


def test_done_flushes_choices_without_finish_reason() -> None:
    outputs = run([chunk(delta(content="ssn 123-45-6789")), SSEEvent(data="[DONE]")])
    assert outputs[-1] == "[DONE]"
    assert text_of(outputs) == f"ssn {R}"


def test_eof_without_done_flushes_held_text() -> None:
    outputs = run([chunk(delta(content="bye bob@example.com"))])
    assert text_of(outputs) == f"bye {R}"


def test_choices_are_redacted_independently() -> None:
    # n=2, interleaved: each choice has half an email in flight at the same time.
    events = [
        chunk(delta(0, content="a: ann@ex")),
        chunk(delta(1, content="b: bob@ex")),
        chunk(delta(0, content="ample.com.")),
        chunk(delta(1, content="ample.org!")),
        chunk(delta(0, finish="stop"), delta(1, finish="stop")),
    ]
    outputs = run(events)
    assert text_of(outputs, 0) == f"a: {R}."
    assert text_of(outputs, 1) == f"b: {R}!"


def test_tool_call_arguments_are_redacted_per_call() -> None:
    first = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": ""}}
    second = {"index": 1, "id": "call_2", "type": "function", "function": {"name": "pay", "arguments": ""}}
    events = [
        chunk(delta(tool_calls=[first, second])),
        chunk(delta(tool_calls=[{"index": 0, "function": {"arguments": '{"email": "jo@ex'}}])),
        chunk(delta(tool_calls=[{"index": 1, "function": {"arguments": '{"card": "4111 1111 '}}])),
        chunk(delta(tool_calls=[{"index": 0, "function": {"arguments": 'ample.com"}'}}])),
        chunk(delta(tool_calls=[{"index": 1, "function": {"arguments": '1111 1111"}'}}])),
        chunk(delta(finish="tool_calls")),
    ]
    outputs = run(events)
    args: dict[int, str] = {0: "", 1: ""}
    for out in outputs:
        for choice in out["choices"]:
            for call in choice["delta"].get("tool_calls", []):
                args[call["index"]] += call.get("function", {}).get("arguments", "")
    assert json.loads(args[0]) == {"email": R}
    assert json.loads(args[1]) == {"card": R}
    assert outputs[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"


def test_refusal_is_redacted() -> None:
    outputs = run([chunk(delta(refusal="I won't email x@y.com")), SSEEvent(data="[DONE]")])
    assert text_of(outputs, field="refusal") == f"I won't email {R}"


def test_non_chunk_payloads_are_redacted_as_text() -> None:
    error = {"error": {"message": "bad input from a@b.com", "code": 400}}
    assert run([SSEEvent(data=json.dumps(error))]) == [
        {"error": {"message": f"bad input from {R}", "code": 400}}
    ]
    assert run([SSEEvent(data="raw 123-45-6789 text")]) == [f"raw {R} text"]


def test_usage_and_comment_events_pass_through() -> None:
    redactor = ChatStreamRedactor()
    keep_alive = SSEEvent(comments=["ping"])
    assert redactor.process(keep_alive) == [keep_alive]
    usage = {**BASE, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}
    (out,) = redactor.process(SSEEvent(data=json.dumps(usage)))
    assert json.loads(out.data) == usage


def test_counts_redactions() -> None:
    redactor = ChatStreamRedactor()
    for event in [chunk(delta(content="a@b.io and 4111111111111111 ")), chunk(delta(content="x@y.io"))]:
        redactor.process(event)
    redactor.finish()
    assert redactor.redactions == 3


def test_non_streaming_completion() -> None:
    body = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Card 4111-1111-1111-1111, mail a.b@c.io",
                    "tool_calls": [
                        {
                            "id": "t",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"s":"123 45 6789"}'},
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
    }
    redacted, count = redact_completion(body)
    message = redacted["choices"][0]["message"]
    assert message["content"] == f"Card {R}, mail {R}"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"s": R}
    assert count == 3
