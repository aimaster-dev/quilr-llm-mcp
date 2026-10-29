"""Incremental SSE parsing: events must come out the same however the bytes are split."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from stream_guardrail.sse import SSEError, SSEEvent, SSEParser

STREAM = (
    'data: {"choices":[{"delta":{"content":"héllo 名前 \U0001f600"}}]}\r\n\r\n'
    ": keep-alive\n\n"
    "event: update\nid: 7\nretry: 1500\ndata: line one\ndata:line two\n\n"
    "data: carriage\r\rdata: [DONE]\n\n"
).encode()

EXPECTED = [
    SSEEvent(data='{"choices":[{"delta":{"content":"héllo 名前 \U0001f600"}}]}'),
    SSEEvent(comments=["keep-alive"]),
    SSEEvent(data="line one\nline two", event="update", id="7", retry="1500"),
    SSEEvent(data="carriage"),
    SSEEvent(data="[DONE]"),
]


def parse(chunks: list[bytes]) -> list[SSEEvent]:
    parser = SSEParser()
    events = [event for chunk in chunks for event in parser.feed(chunk)]
    return events + parser.close()


def test_parses_a_whole_stream() -> None:
    assert parse([STREAM]) == EXPECTED


def test_every_single_split_point() -> None:
    # Includes splits inside multi-byte UTF-8 characters and between "\r" and "\n".
    for i in range(len(STREAM) + 1):
        assert parse([STREAM[:i], STREAM[i:]]) == EXPECTED, i


def test_byte_by_byte() -> None:
    assert parse([STREAM[i : i + 1] for i in range(len(STREAM))]) == EXPECTED


@given(st.lists(st.integers(0, len(STREAM)), max_size=12))
def test_random_splits(cuts: list[int]) -> None:
    bounds = [0, *sorted(cuts), len(STREAM)]
    assert parse([STREAM[a:b] for a, b in itertools.pairwise(bounds)]) == EXPECTED


def test_crlf_split_across_chunks_is_one_line_ending() -> None:
    # If "\r" and "\n" were treated as two line endings, the empty line in between would
    # dispatch "a" early and "b" as a separate event.
    assert parse([b"data: a\r", b"\ndata: b\r\n\r\n"]) == [SSEEvent(data="a\nb")]


def test_lone_cr_in_the_middle_does_not_swallow_the_next_newline() -> None:
    assert parse([b"data: a\rdata: b", b"\n\n"]) == [SSEEvent(data="a\nb")]


def test_events_are_emitted_as_soon_as_they_are_complete() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: one\n") == []
    assert parser.feed(b"\ndata: tw") == [SSEEvent(data="one")]
    assert parser.feed(b"o\n\n") == [SSEEvent(data="two")]


def test_field_parsing_rules() -> None:
    events = parse([b"data\n\ndata:\n\ndata:  two spaces\nfoo: ignored\n\n"])
    assert events == [SSEEvent(data=""), SSEEvent(data=""), SSEEvent(data=" two spaces")]


def test_unterminated_event_is_dispatched_at_eof() -> None:
    assert parse([b"data: tail"]) == [SSEEvent(data="tail")]


def test_invalid_utf8_is_replaced_not_fatal() -> None:
    assert parse([b"data: \xff\n\n"]) == [SSEEvent(data="�")]


def test_oversized_event_is_rejected() -> None:
    parser = SSEParser(max_event_size=100)
    with pytest.raises(SSEError):
        parser.feed(b"data: " + b"x" * 200)
    parser = SSEParser(max_event_size=100)
    with pytest.raises(SSEError):
        for _ in range(30):
            parser.feed(b"data: xxxxxx\n")


def test_encode_round_trip() -> None:
    for event in EXPECTED:
        assert parse([event.encode()]) == [event]
    assert SSEEvent(data="a\nb").encode() == b"data: a\ndata: b\n\n"
