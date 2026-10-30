"""Incremental SSE parsing: events must come out the same however the bytes are split."""

from __future__ import annotations

import codecs
import itertools
import time

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


BOM = codecs.BOM_UTF8  # EF BB BF


def test_a_leading_bom_is_stripped_wherever_the_bytes_split() -> None:
    stream = BOM + STREAM
    assert parse([stream]) == EXPECTED
    for i in range(len(BOM) + 4):  # splits inside the BOM's three bytes and just after it
        assert parse([stream[:i], stream[i:]]) == EXPECTED, i
    assert parse([stream[i : i + 1] for i in range(len(stream))]) == EXPECTED


def test_only_one_leading_bom_is_stripped() -> None:
    # A second BOM, or one later in the stream, is part of a field name (U+FEFF then "data"),
    # which is unknown, so the line is ignored as the spec says.
    assert parse([BOM + BOM + b"data: a\n\n"]) == []
    assert parse([b"data: a\n\n" + BOM + b"data: b\n\n"]) == [SSEEvent(data="a")]


def test_a_long_line_in_small_reads_costs_linear_time() -> None:
    # 900 KB in 256-byte reads (3,600 reads). Rebuilding and rescanning the partial line on every
    # read took ~10 s here; linear parsing takes milliseconds. The bound is generous for slow CI.
    payload = b"data: " + b"x" * (900 << 10) + b"\r"
    reads = [payload[i : i + 256] for i in range(0, len(payload), 256)]
    parser = SSEParser()
    started = time.perf_counter()
    events = [event for read in reads for event in parser.feed(read)]
    events += parser.feed(b"\n\r\n") + parser.close()  # the CRLF is split across reads
    elapsed = time.perf_counter() - started
    assert events == [SSEEvent(data="x" * (900 << 10))]
    assert elapsed < 2.0, f"{elapsed:.2f} s"


def test_a_line_split_across_many_reads_is_joined_once_and_correctly() -> None:
    parser = SSEParser()
    assert parser.feed(b"da") == []
    assert parser.feed(b"ta: hel") == []
    assert parser.feed(b"lo\r") == []
    assert parser.feed(b"\ndata: wor") == []  # "\r" + "\n" is one line ending
    assert parser.feed(b"ld\n\n") == [SSEEvent(data="hello\nworld")]


def test_events_before_an_oversized_event_are_kept() -> None:
    # One read carries two good events and the start of an oversized one.
    parser = SSEParser(max_event_size=100)
    with pytest.raises(SSEError) as caught:
        parser.feed(b"data: one\n\ndata: two\n\ndata: " + b"x" * 200)
    assert caught.value.events == [SSEEvent(data="one"), SSEEvent(data="two")]
    # The same when the limit is crossed by complete lines of one event.
    parser = SSEParser(max_event_size=100)
    with pytest.raises(SSEError) as caught:
        parser.feed(b": ping\n\n" + b"data: xxxxxxxxxx\n" * 20)
    assert caught.value.events == [SSEEvent(comments=["ping"])]


def test_a_failed_parser_stays_failed() -> None:
    parser = SSEParser(max_event_size=10)
    with pytest.raises(SSEError):
        parser.feed(b"data: " + b"x" * 20)
    with pytest.raises(SSEError) as caught:
        parser.feed(b"\n\ndata: ok\n\n")
    assert caught.value.events == []
    with pytest.raises(SSEError):
        parser.close()


def test_encode_round_trip() -> None:
    for event in EXPECTED:
        assert parse([event.encode()]) == [event]
    assert SSEEvent(data="a\nb").encode() == b"data: a\ndata: b\n\n"
