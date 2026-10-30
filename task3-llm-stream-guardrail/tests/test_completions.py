"""Redaction of OpenAI chat-completion chunks and responses."""

from __future__ import annotations

import itertools
import json
import random
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from stream_guardrail.completions import ChatStreamRedactor, redact_completion
from stream_guardrail.redaction import REDACTED, redact
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


IDENTITY_KEYS = {"index", "type", "id", "role", "name"}


def accumulate(target: Any, delta: Any) -> Any:
    """Merge a delta into the accumulated message the way OpenAI client libraries do.

    Text is concatenated, identity values (``id``, ``role``, ``function.name``...) are kept from
    their first delta, tool calls are matched by ``index`` and other list elements by position.
    """
    if isinstance(delta, dict):
        target = target if isinstance(target, dict) else {}
        for key, value in delta.items():
            if isinstance(value, str) and key in IDENTITY_KEYS:
                target.setdefault(key, value)
            else:
                target[key] = accumulate(target.get(key), value)
        return target
    if isinstance(delta, list):
        target = target if isinstance(target, list) else []
        for position, item in enumerate(delta):
            index = item.get("index") if isinstance(item, dict) else None
            if isinstance(index, int):
                found = next((t for t in target if isinstance(t, dict) and t.get("index") == index), None)
                if found is None:
                    target.append(accumulate(None, item))
                else:
                    accumulate(found, item)
            elif position < len(target):
                target[position] = accumulate(target[position], item)
            else:
                target.append(accumulate(None, item))
        return target
    if isinstance(delta, str):
        return (target if isinstance(target, str) else "") + delta
    return delta


def messages_of(outputs: list[Any]) -> dict[int, Any]:
    """What a client assembles from the output: one accumulated delta per choice index."""
    messages: dict[int, Any] = {}
    for out in outputs:
        for choice in out.get("choices", []) if isinstance(out, dict) else []:
            messages[choice["index"]] = accumulate(messages.get(choice["index"]), choice.get("delta"))
    return messages


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


def test_nothing_held_means_no_synthetic_chunk() -> None:
    outputs = run([chunk(delta(content="hi! ", reasoning="ok. ")), SSEEvent(data="[DONE]")])
    assert len(outputs) == 2
    assert outputs[0]["choices"][0]["delta"] == {"content": "hi! ", "reasoning": "ok. "}
    assert outputs[1] == "[DONE]"


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


# --------------------------------------------------------------------------------------------- #
# Deny by default: every model-written string, in every field, streaming and not
# --------------------------------------------------------------------------------------------- #
PIECES = ["Write to john", ".doe@exa", "mple.com or 4111 ", "1111 1111 1111", " now"]
EXPECTED = f"Write to {R} or {R} now"


def tool_call(index: int, first: bool, arguments: str) -> dict[str, Any]:
    call: dict[str, Any] = {"index": index, "function": {"arguments": arguments}}
    if first:
        call.update(id=f"call_{index}", type="function")
        call["function"]["name"] = "lookup"
    return {"tool_calls": [call]}


# name -> (delta builder taking (is first delta, text), path to the text in the client's message).
# A path step ("index", k) selects the list element whose "index" is k.
FIELDS: dict[str, tuple[Any, tuple[Any, ...]]] = {
    "refusal": (lambda first, t: {"refusal": t}, ("refusal",)),
    "reasoning_content": (lambda first, t: {"reasoning_content": t}, ("reasoning_content",)),
    "reasoning": (lambda first, t: {"reasoning": t}, ("reasoning",)),
    "audio.transcript": (
        lambda first, t: {"audio": {"id": "audio_1", "transcript": t} if first else {"transcript": t}},
        ("audio", "transcript"),
    ),
    "content parts": (
        lambda first, t: {"content": [{"type": "text", "text": t}]},
        ("content", 0, "text"),
    ),
    "tool call arguments": (
        lambda first, t: tool_call(0, first, t),
        ("tool_calls", ("index", 0), "function", "arguments"),
    ),
    "legacy function_call": (
        lambda first, t: {"function_call": {"name": "lookup", "arguments": t} if first else {"arguments": t}},
        ("function_call", "arguments"),
    ),
    "unknown field": (lambda first, t: {"x_vendor": {"notes": [t]}}, ("x_vendor", "notes", 0)),
}


def dig(value: Any, path: tuple[Any, ...]) -> Any:
    for step in path:
        if isinstance(step, tuple):
            value = next(e for e in value if e.get("index") == step[1])
        else:
            value = value[step]
    return value


def field_events(build: Any, pieces: list[str], index: int = 0) -> list[SSEEvent]:
    return [
        chunk({"index": index, "delta": build(i == 0, piece), "finish_reason": None})
        for i, piece in enumerate(pieces)
    ]


@pytest.mark.parametrize("finish", [True, False], ids=["finish_reason", "done-only"])
@pytest.mark.parametrize("name", list(FIELDS))
def test_every_delta_field_is_redacted_and_flushed(name: str, finish: bool) -> None:
    build, path = FIELDS[name]
    events = [chunk(delta(role="assistant")), *field_events(build, PIECES)]
    if finish:
        events.append(chunk(delta(finish="stop")))
    outputs = run([*events, SSEEvent(data="[DONE]")])
    assert dig(messages_of(outputs)[0], path) == EXPECTED
    raw = json.dumps(outputs)
    assert "john" not in raw
    assert "1111" not in raw
    assert outputs[-1] == "[DONE]"
    if finish:
        # Every character arrived before the finish_reason chunk, which carries no text.
        assert outputs[-2]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]


def test_structural_values_are_neither_redacted_nor_held() -> None:
    # Not text the model wrote: a tool call's id and function name must arrive intact, at once.
    call = {
        "index": 0,
        "id": "call_4111111111111111",
        "type": "function",
        "function": {"name": "lookup_customer", "arguments": ""},
    }
    (out,) = ChatStreamRedactor().process(chunk(delta(role="assistant", tool_calls=[call])))
    (choice,) = json.loads(out.data or "")["choices"]
    assert choice["delta"] == {"role": "assistant", "tool_calls": [call]}


def test_tool_calls_are_keyed_by_their_index_not_their_position() -> None:
    first = {"tool_calls": [
        {"index": 0, "function": {"arguments": '{"to": "ann@ex'}},
        {"index": 1, "function": {"arguments": '{"card": "4111 1111 '}},
    ]}  # fmt: skip
    events = [
        chunk(delta(**first)),
        # Call 1 is now alone, at position 0, which call 0 occupied before.
        chunk(delta(**tool_call(1, False, '1111 1111"}'))),
        chunk(delta(**tool_call(0, False, 'ample.com"}'))),
        chunk(delta(finish="tool_calls")),
    ]
    calls = messages_of(run(events))[0]["tool_calls"]
    assert {c["index"]: json.loads(c["function"]["arguments"]) for c in calls} == {
        0: {"to": R},
        1: {"card": R},
    }


def test_logprobs_from_upstream_are_dropped() -> None:
    def token(t: str) -> dict[str, Any]:
        return {"token": t, "logprob": -0.1, "bytes": list(t.encode()), "top_logprobs": []}

    events = [
        chunk(
            {"index": 0, "delta": {"content": t}, "logprobs": {"content": [token(t)]}, "finish_reason": None}
        )
        for t in ["Mail ", "john", "@example", ".com", " now"]
    ]
    outputs = run([*events, chunk(delta(finish="stop")), SSEEvent(data="[DONE]")])
    assert text_of(outputs) == f"Mail {R} now"
    assert "john" not in json.dumps(outputs)
    assert all(c.get("logprobs") is None for out in outputs[:-1] for c in out["choices"])


def test_other_choice_fields_are_redacted_one_shot() -> None:
    # A legacy completions-style "text" field holds complete values, so it is not held back.
    (out,) = ChatStreamRedactor().process(chunk({"index": 0, "text": "mail a@b.io", "finish_reason": None}))
    assert json.loads(out.data or "")["choices"][0]["text"] == f"mail {R}"


def content_text(outputs: list[Any]) -> str:
    texts = []
    for out in outputs:
        for choice in out["choices"]:
            content = choice["delta"].get("content")
            texts += [content] if isinstance(content, str) else [p.get("text", "") for p in content or []]
    return "".join(texts)


def test_content_string_and_text_parts_are_one_stream() -> None:
    # Clients read both shapes as the message text, so an email split across them is caught.
    events = [
        chunk(delta(content="mail ann@ex")),
        chunk(
            delta(content=[{"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "am"}])
        ),
        chunk(delta(content=[{"type": "text", "text": "ple.com"}], finish="stop")),
    ]
    outputs = run(events)
    assert content_text(outputs) == f"mail {R}"
    assert outputs[-1]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}


def test_a_tail_that_does_not_fit_the_finish_delta_gets_a_chunk_of_its_own() -> None:
    events = [
        chunk(delta(x_vendor="mail ann@example.com")),
        chunk(delta(x_vendor={"note": "ok"}, finish="stop")),  # same field, now an object
    ]
    *body, finish = run(events)
    assert finish["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    pieces = [c["delta"]["x_vendor"] for out in body for c in out["choices"]]
    assert pieces == ["mail ", {"note": "ok"}, R]


def test_non_streaming_message_fields_are_all_redacted() -> None:
    secret = "ann@corp.io"
    body = {
        "id": "chatcmpl-3",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"part {secret}"}],
                    "refusal": f"no {secret}",
                    "reasoning_content": f"think {secret}",
                    "reasoning": f"reason {secret}",
                    "audio": {"id": "audio_1", "data": "UklGRg==", "transcript": f"say {secret}"},
                    "tool_calls": [
                        {
                            "id": "call_4111111111111111",
                            "type": "function",
                            "function": {"name": "lookup_customer", "arguments": f'{{"to":"{secret}"}}'},
                        }
                    ],
                    "x_vendor": {"notes": [f"note {secret}"]},
                },
                "logprobs": {
                    "content": [{"token": secret, "logprob": -0.1, "bytes": [], "top_logprobs": []}]
                },
                "finish_reason": "stop",
            }
        ],
    }
    redacted, count = redact_completion(body)
    assert secret not in json.dumps(redacted)
    choice = redacted["choices"][0]
    message = choice["message"]
    assert choice["logprobs"] is None
    assert choice["finish_reason"] == "stop"
    assert message["role"] == "assistant"
    assert message["content"] == [{"type": "text", "text": f"part {R}"}]
    assert message["refusal"] == f"no {R}"
    assert message["reasoning_content"] == f"think {R}"
    assert message["reasoning"] == f"reason {R}"
    assert message["audio"] == {"id": "audio_1", "data": "UklGRg==", "transcript": f"say {R}"}
    assert message["tool_calls"][0]["id"] == "call_4111111111111111"
    assert message["tool_calls"][0]["function"] == {"name": "lookup_customer", "arguments": f'{{"to":"{R}"}}'}
    assert message["x_vendor"] == {"notes": [f"note {R}"]}
    assert count == 7


# --------------------------------------------------------------------------------------------- #
# Property: whatever the interleaving of fields and choices, each field reads as one-shot
# redaction of its text, and nothing arrives after its choice's finish_reason
# --------------------------------------------------------------------------------------------- #
FIELD_TEXT = st.lists(
    st.sampled_from(["john.doe@example.com", "4111 1111 1111 1111", "123-45-6789", "a@b.co", " ", "\n"])
    | st.text(alphabet="ab @.-0123456789", max_size=6),
    max_size=6,
).map("".join)

# (choice index, delta builder, path). Tool calls 0 and 3 always sit at list position 0.
STREAMS = [
    (0, *FIELDS["refusal"]),
    (0, *FIELDS["reasoning_content"]),
    (0, *FIELDS["audio.transcript"]),
    (0, *FIELDS["tool call arguments"]),
    (0, lambda first, t: tool_call(3, first, t), ("tool_calls", ("index", 3), "function", "arguments")),
    (1, *FIELDS["content parts"]),
    (1, *FIELDS["reasoning"]),
]


@st.composite
def interleaved_streams(draw: st.DrawFn) -> tuple[list[str], list[SSEEvent], str]:
    texts = [draw(FIELD_TEXT) for _ in STREAMS]
    pieces = []
    for text in texts:
        cuts = sorted(draw(st.lists(st.integers(0, len(text)), max_size=4)))
        pieces.append([text[a:b] for a, b in itertools.pairwise([0, *cuts, len(text)]) if b > a])
    rng: random.Random = draw(st.randoms(use_true_random=False))
    order = [i for i, field_pieces in enumerate(pieces) for _ in field_pieces]
    rng.shuffle(order)  # the k-th occurrence of field i sends its k-th piece
    sent = [0] * len(STREAMS)
    events = []
    for i in order:
        index, build, _ = STREAMS[i]
        piece = pieces[i][sent[i]]
        events.append(chunk({"index": index, "delta": build(sent[i] == 0, piece), "finish_reason": None}))
        sent[i] += 1
    ending = draw(st.sampled_from(["finish", "done", "eof"]))
    if ending == "finish":
        events += [chunk(delta(i, finish="stop")) for i in rng.sample([0, 1], 2)]
    if ending != "eof":
        events.append(SSEEvent(data="[DONE]"))
    return texts, events, ending


@given(interleaved_streams())
def test_any_interleaving_of_fields_and_choices_reads_like_one_shot(
    case: tuple[list[str], list[SSEEvent], str],
) -> None:
    texts, events, ending = case
    outputs = run(events)
    messages = messages_of(outputs)
    for (index, _, path), text in zip(STREAMS, texts, strict=True):
        if text:
            assert dig(messages[index], path) == redact(text)
    finished: set[int] = set()
    for out in outputs:
        for choice in out["choices"] if isinstance(out, dict) else []:
            assert choice["index"] not in finished, "text after finish_reason"
            if choice.get("finish_reason") is not None:
                finished.add(choice["index"])
    assert finished == ({0, 1} if ending == "finish" else set())
    assert (outputs[-1] == "[DONE]") == (ending != "eof")
