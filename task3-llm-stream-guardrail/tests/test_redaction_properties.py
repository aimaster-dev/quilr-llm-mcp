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
    "jose\u0301@example.com",
    "请联系",
    "获取帮助",
    "git@github.com:",
    "icon@2x.png",
    "4111111111111111",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "4111\u20111111\u20111111\u20111111",
    "4111\u00a01111\u00a01111\u00a01111",
    "4111\u202f1111\u20091111\u20071111",
    "4111 1111 1111 1111 110",
    "3782 822463 10005",
    "3056 930902 5904",
    "4222222222222",
    "6011000990139420007",
    "4111111111111112",
    "555 0100 ",
    "2 3 5 7 11 13 17 19 23 29 31",
    "2024-01-15 ",
    "123-45-6789",
    "123 45 6789",
    "123\u201145\u20116789",
    "123\u00a045\u00a06789",
    "123-45\u00a06789",
    "000-12-3456",
    "666-12-3456",
    "123-45-67890",
    "a" * 70,
    "7" * 25,
    "4" * 25,
    "-",
    "\u2011",
    "\u2013",
    " ",
    "\u00a0",
    "  ",
    "\n",
]
# Single characters that sit on every boundary: separators of both kinds, CJK, a combining mark.
NOISE = st.text(alphabet="ab xyz0123456789 -.@_+%:/\né名获\u2011\u2013\u00a0\u202f\u3000\u0301", max_size=12)
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


SAFE_WORDS = st.sampled_from(
    [
        "hello",
        "the",
        "2024",
        "v1.2",
        "a-b",
        "x_y",
        "50%",
        "名前",
        "1234 5678",
        "2024-01-15 2024-02-15",
        "call 555 0100",
        "2 3 5 7 11 13 17 19 23 29 31",
        "1234567890123452",
        "icon@2x.png",
        "git@github.com:org/repo.git",
    ]
)
R = REDACTED
# (text, what one-shot redaction must turn it into)
PII_WORDS = st.sampled_from(
    [
        ("john.doe@example.com", R),
        ("jürgen@exämple.de", R),
        ("jose\u0301@example.com", R),
        ("请联系john@example.com获取", f"请联系{R}获取"),
        ("4111 1111 1111 1111", R),
        ("4111-1111-1111-1111", R),
        ("4111\u20111111\u20111111\u20111111", R),
        ("4111\u00a01111\u00a01111\u00a01111", R),
        ("3056 930902 5904", R),
        ("378282246310005", R),
        ("123-45-6789", R),
        ("123\u201145\u20116789", R),
        ("078\u00a005\u00a01120", R),
        ("078 05 1120", R),
    ]
)
SEPARATORS = st.sampled_from([", ", "; ", "\n", " and ", " | "])


@given(st.lists(st.tuples(st.booleans(), SAFE_WORDS, PII_WORDS), max_size=12), SEPARATORS)
def test_inserted_pii_never_leaks(words: list[tuple[bool, str, tuple[str, str]]], sep: str) -> None:
    chosen = [pii[0] if use_pii else safe for use_pii, safe, pii in words]
    expected = sep.join(pii[1] if use_pii else safe for use_pii, safe, pii in words)
    text = sep.join(chosen)
    redactor = StreamRedactor()
    streamed = "".join(redactor.feed(text[i : i + 3]) for i in range(0, len(text), 3)) + redactor.flush()
    assert redact(text) == expected
    assert streamed == expected
