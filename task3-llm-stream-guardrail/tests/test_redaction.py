"""Detector semantics (one-shot) and streaming behaviour of the redaction engine."""

from __future__ import annotations

import itertools

import pytest

from stream_guardrail.redaction import (
    LOCAL_MAX,
    MAX_BUFFERED,
    REDACTED,
    StreamRedactor,
    redact,
)

R = REDACTED


def stream(chunks: list[str]) -> str:
    redactor = StreamRedactor()
    return "".join(redactor.feed(chunk) for chunk in chunks) + redactor.flush()


# --------------------------------------------------------------------------------------------- #
# One-shot semantics
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mail john.doe@example.com.", f"Mail {R}."),
        ("a.b+tag@sub.example.co.uk", R),
        ("<john@example.com>", f"<{R}>"),
        ("mailto:jane_doe99@mail.example.org?subject=hi", f"mailto:{R}?subject=hi"),
        ("JOHN@EXAMPLE.COM", R),
        ("jürgen@exämple.de", R),
        ("x@site.xn--p1ai ok", f"{R} ok"),
        ("4111111111111111@example.com", R),  # the email wins over the card it contains
        # A trailing label that cannot be a TLD is not part of the address.
        ("john@example.com.123", f"{R}.123"),
        # Not emails: no dot in the domain, no local part, single-letter TLD.
        ("foo@bar", "foo@bar"),
        ("user@localhost", "user@localhost"),
        ("@example.com", "@example.com"),
        ("a@b.c", "a@b.c"),
        # Local parts are capped at 64 chars, so a longer run keeps its head. The domain is always
        # redacted.
        ("a" * 70 + "@x.com", "a" * 6 + R),
    ],
)
def test_emails(text: str, expected: str) -> None:
    assert redact(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4111111111111111", R),
        ("card: 4111 1111 1111 1111.", f"card: {R}."),
        ("4111-1111-1111-1111", R),
        ("3782 822463 10005", R),  # Amex 4-6-5 grouping
        ("4222222222222", R),  # 13 digits
        ("6011000990139420007", R),  # 19 digits
        ("x4111111111111111", f"x{R}"),
        ("4111111111111111abc", f"{R}abc"),
        # Luhn failures are left alone.
        ("4111111111111112", "4111111111111112"),
        ("order 1234567890123", "order 1234567890123"),
        # Digits glued to the number make it a different, longer number.
        ("14111111111111111", "14111111111111111"),
        ("12345678901234567890", "12345678901234567890"),
        # The longest Luhn-valid run that ends on a group boundary wins, so a trailing group
        # does not hide the card.
        ("4111 1111 1111 1111 2 times", f"{R} 2 times"),
        ("1 4111 1111 1111 1111", f"1 {R}"),
        # Separators must be single spaces or dashes.
        ("4111  1111  1111  1111", "4111  1111  1111  1111"),
        ("4111--1111--1111--1111", "4111--1111--1111--1111"),
    ],
)
def test_payment_cards(text: str, expected: str) -> None:
    assert redact(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SSN 123-45-6789.", f"SSN {R}."),
        ("123 45 6789", R),
        ("078-05-1120", R),
        # Never-issued ranges.
        ("000-12-3456", "000-12-3456"),
        ("666-12-3456", "666-12-3456"),
        ("900-12-3456", "900-12-3456"),
        ("123-00-4567", "123-00-4567"),
        ("123-45-0000", "123-45-0000"),
        # Mixed separators, glued digits and bare 9-digit numbers are not SSNs.
        ("123-45 6789", "123-45 6789"),
        ("1123-45-6789", "1123-45-6789"),
        ("123-45-67890", "123-45-67890"),
        ("123456789", "123456789"),
    ],
)
def test_ssns(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_multiple_and_adjacent_matches() -> None:
    text = "a@b.co,123-45-6789;4111111111111111 x@y.io"
    assert redact(text) == f"{R},{R};{R} {R}"


def test_custom_replacement() -> None:
    assert redact("ssn 123-45-6789", replacement="***") == "ssn ***"
    redactor = StreamRedactor(replacement="<pii>")
    assert redactor.feed("ssn 123-45-6789 ") + redactor.flush() == "ssn <pii> "


def test_plain_text_is_untouched() -> None:
    text = "Version 2.0 shipped on 2024-05-01 at 10:30. Call 555-0100 or visit example.com!"
    assert redact(text) == text


# --------------------------------------------------------------------------------------------- #
# Streaming behaviour
# --------------------------------------------------------------------------------------------- #
def test_email_split_across_chunks() -> None:
    redactor = StreamRedactor()
    assert redactor.feed("Contact john.doe@ex") == "Contact "
    assert redactor.feed("ample.com today") == f"{R} "
    assert redactor.flush() == "today"
    assert redactor.redactions == 1


def test_prose_flows_through_immediately() -> None:
    redactor = StreamRedactor()
    assert redactor.feed("Hello there, ") == "Hello there, "
    assert redactor.feed("how are") == "how "  # "are" could still become are@x.com
    assert redactor.pending == "are"
    assert redactor.feed(" you?\n") == "are you?\n"
    assert redactor.pending == ""


def test_complete_match_touching_the_end_is_held() -> None:
    # "a@b.co" is already a valid email but "a@b.com" would be a longer one.
    redactor = StreamRedactor()
    assert redactor.feed("x a@b.co") == "x "
    assert redactor.feed("m") == ""
    assert redactor.feed(" y") == f"{R} "
    assert redactor.flush() == "y"


def test_card_split_digit_by_digit() -> None:
    assert stream(list("pay 4111 1111 1111 1111 now")) == f"pay {R} now"


def test_rejected_candidate_is_released_once_decided() -> None:
    redactor = StreamRedactor()
    assert redactor.feed("id 4111111111111112") == "id "
    assert redactor.feed(".") == ""  # "4111111111111112.x@y.com" would still be an email
    assert redactor.feed(",") == "4111111111111112.,"


def test_flush_releases_the_held_tail_and_resets() -> None:
    redactor = StreamRedactor()
    assert redactor.feed("ends with john@example.com") == "ends with "
    assert redactor.flush() == R
    assert redactor.flush() == ""
    assert redactor.buffered == 0
    assert redactor.feed("again 123-45-6789\n") == f"again {R}\n"


def test_empty_chunks_are_no_ops() -> None:
    redactor = StreamRedactor()
    assert redactor.feed("") == ""
    assert redactor.feed("abc") == ""
    assert redactor.feed("") == ""
    assert redactor.flush() == "abc"


def test_long_word_run_is_not_held_forever() -> None:
    # Without an "@", only the last LOCAL_MAX characters could still be part of an email.
    redactor = StreamRedactor()
    emitted = redactor.feed("a" * 10_000)
    assert len(redactor.pending) == LOCAL_MAX
    assert emitted == "a" * (10_000 - LOCAL_MAX)


def test_long_digit_run_is_not_held_forever() -> None:
    redactor = StreamRedactor()
    redactor.feed("7" * 10_000)
    assert len(redactor.pending) <= LOCAL_MAX


def test_buffer_stays_bounded_on_a_long_stream() -> None:
    redactor = StreamRedactor()
    piece = "user.name@mail.example.com 4111-1111-1111-1111 lorem ipsum 123-45-6789 "
    peak = 0
    for i in range(5_000):
        redactor.feed(piece[i % 7 :] + piece[: i % 7])
        peak = max(peak, redactor.buffered)
    assert peak <= MAX_BUFFERED
    assert peak < 100  # in practice only the last word or number is held


@pytest.mark.parametrize(
    "text",
    [
        "Contact john.doe@example.com or 4111 1111 1111 1111!",
        "x a@b.co%x@y.com",
        "ssn:123-45-6789,123 45 6789 and 123-45-67890",
        "4111 1111 1111 1111 2 and 1 4111 1111 1111 1111",
        "abc@def@ghi.jk lm@no.p.qr",
        "a@b.c.d.e.f.g.h.i.j.com x@y.zz",
        "john@example.com.123 jürgen@exämple.de",
        "41111111111111105 4222222222222-1",
    ],
)
def test_every_two_and_three_way_split_matches_one_shot(text: str) -> None:
    expected = redact(text)
    cuts = range(len(text) + 1)
    for i in cuts:
        assert stream([text[:i], text[i:]]) == expected
    for i, j in itertools.combinations(cuts, 2):
        assert stream([text[:i], text[i:j], text[j:]]) == expected
    assert stream(list(text)) == expected
