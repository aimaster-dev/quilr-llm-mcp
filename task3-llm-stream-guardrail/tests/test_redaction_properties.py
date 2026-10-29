"""Property-based tests: streaming must be indistinguishable from one-shot redaction."""

from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st

from stream_guardrail.redaction import MAX_BUFFERED, REDACTED, StreamRedactor, redact

# Fragments that exercise every detector, its validators and its boundaries. Concatenated without
# separators they also create the awkward adjacencies (digits glued to cards, dots after domains...).
PII_FRAGMENTS = [
    "john.doe@example.com",
    "a@b.co",
    "x@y.io.",
    "jürgen@exämple.de",
    "name+tag@mail.example.co.uk",
    "@example.com",
    "@b.c",
    ".com",
    ".123",
    "xn--p1ai",
    "4111111111111111",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "3782 822463 10005",
    "4222222222222",
    "6011000990139420007",
    "4111111111111112",
    "123-45-6789",
    "123 45 6789",
    "000-12-3456",
    "666-12-3456",
    "123-45-67890",
    "a" * 70,
    "7" * 25,
    "-",
    " ",
    "  ",
    "\n",
]
NOISE = st.text(alphabet="ab xyz0123456789 -.@_+%\né名", max_size=12)
TEXTS = st.lists(st.one_of(st.sampled_from(PII_FRAGMENTS), NOISE), max_size=14).map("".join)


@st.composite
def text_and_chunks(draw: st.DrawFn) -> tuple[str, list[str]]:
    text = draw(TEXTS)
    # Sorted cut points with repeats, so chunks can be empty, single characters or the whole text.
    cuts = sorted(draw(st.lists(st.integers(0, len(text)), max_size=len(text) + 2)))
    bounds = [0, *cuts, len(text)]
    return text, [text[a:b] for a, b in itertools.pairwise(bounds)]


@given(text_and_chunks())
def test_chunking_invariance(case: tuple[str, list[str]]) -> None:
    text, chunks = case
    redactor = StreamRedactor()
    out = []
    for chunk in chunks:
        out.append(redactor.feed(chunk))
        assert redactor.buffered <= MAX_BUFFERED
    out.append(redactor.flush())
    assert "".join(out) == redact(text)


@given(TEXTS)
def test_single_character_chunks(text: str) -> None:
    redactor = StreamRedactor()
    streamed = "".join(redactor.feed(ch) for ch in text) + redactor.flush()
    assert streamed == redact(text)


@given(TEXTS)
def test_nothing_is_held_after_a_newline(text: str) -> None:
    # A newline cannot be part of any detector's match, so the stream must not hold text across one.
    redactor = StreamRedactor()
    emitted = redactor.feed(text + "\n")
    assert redactor.pending == ""
    assert emitted == redact(text + "\n")


SAFE_WORDS = st.sampled_from(["hello", "the", "2024", "v1.2", "a-b", "x_y", "50%", "名前", "1234 5678"])
PII_WORDS = st.sampled_from(
    [
        "john.doe@example.com",
        "jürgen@exämple.de",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "378282246310005",
        "123-45-6789",
        "078 05 1120",
    ]
)
SEPARATORS = st.sampled_from([", ", "; ", "\n", " and ", " | "])


@given(st.lists(st.tuples(st.booleans(), SAFE_WORDS, PII_WORDS), max_size=12), SEPARATORS)
def test_inserted_pii_never_leaks(words: list[tuple[bool, str, str]], sep: str) -> None:
    chosen = [pii if use_pii else safe for use_pii, safe, pii in words]
    expected = sep.join(REDACTED if use_pii else safe for use_pii, safe, _ in words)
    text = sep.join(chosen)
    redactor = StreamRedactor()
    streamed = "".join(redactor.feed(text[i : i + 3]) for i in range(0, len(text), 3)) + redactor.flush()
    assert redact(text) == expected
    assert streamed == expected
