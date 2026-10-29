"""Incremental PII redaction for text that arrives in arbitrary chunks.

Semantics (one shot)
--------------------
``redact(text)`` scans left to right. At each position it tries the detectors in priority order
(email, payment card, US SSN). The first one that matches *and* passes its validator is replaced by
``[REDACTED]`` and scanning resumes after it. If none applies, scanning moves one character right.

Streaming
---------
``StreamRedactor`` runs exactly the same procedure (``_redact_span`` and ``_decide`` below) over a
small rolling buffer. The only extra step is the *hold point*: the leftmost position whose suffix
could still grow into a match. A position before the hold point can be decided now, because no
detector can read past the end of the buffer from there:

* each detector regex only consumes characters from a small alphabet (``_EMAIL_TAIL_REV`` and
  ``_DIGIT_TAIL_REV`` describe the strings it can consume), and every quantifier is bounded;
* the quantifiers are possessive, so the regexes never backtrack, and the validators look at most
  one character past the match.

If a detector's search from ``s`` reached the end of the buffer, the text from ``s`` to the end would
itself be a viable prefix, and ``s`` would be at or after the hold point. So every decision made
before the hold point is the decision one-shot redaction makes on the full text. Text from the hold
point onwards is kept back until more input arrives or the stream ends. This gives the two
invariants the tests check:

* **Chunking invariance**: for any split of a text into chunks, the concatenated output of
  ``feed()``/``flush()`` equals ``redact(text)``.
* **Bounded memory**: the buffer never holds more than ``MAX_BUFFERED`` characters after
  ``feed()`` returns, whatever the response length.

Why not ``regex`` partial matching?  ``regex.search(..., partial=True)`` prefers a complete match
over an earlier partial one (``a.{5}|b`` on ``"a12b"`` returns ``b``). It also treats the end of the
buffer as the end of the text when it evaluates lookaheads. Neither is safe for deciding that a
prefix is final. Deriving the hold point from the detector alphabets is exact for the cases that
matter, needs no third-party engine, and costs O(length of the trailing run).
"""

from __future__ import annotations

import re
from collections.abc import Callable

REDACTED = "[REDACTED]"

# ---------------------------------------------------------------------------------------------
# Bounds. Every quantifier is bounded, so the longest possible match and the longest viable
# prefix (and therefore the hold-back buffer) are bounded too.
# ---------------------------------------------------------------------------------------------
LOCAL_MAX = 64  # RFC 5321 local-part limit
LABEL_MAX = 63  # RFC 1035 label limit
MAX_LABELS = 8  # host labels including the TLD, e.g. mail.eu.corp.example.co.uk is 6
DOMAIN_MAX = MAX_LABELS * LABEL_MAX + (MAX_LABELS - 1)
CARD_MIN_DIGITS = 13
CARD_MAX_DIGITS = 19

# Characters a detector can consume. ``\w`` is Unicode-aware, so "jürgen@exämple.de" is covered.
_LOCAL = r"[\w.%+-]"
_LABEL = rf"[\w-]{{1,{LABEL_MAX}}}+"
_DIGITS = frozenset("0123456789")

EMAIL_PATTERN = rf"{_LOCAL}{{1,{LOCAL_MAX}}}+@{_LABEL}(?:\.{_LABEL}){{1,{MAX_LABELS - 1}}}+"
# 13-19 digits; a single space or dash may separate any two digits ("4111 1111 1111 1111").
CARD_PATTERN = rf"(?<![0-9])[0-9](?:[ -]?[0-9]){{{CARD_MIN_DIGITS - 1},{CARD_MAX_DIGITS - 1}}}+"
# AAA-GG-SSSS or AAA GG SSSS with a consistent separator. Never-issued numbers are excluded:
# area 000, 666 and 900-999, group 00, serial 0000.
SSN_PATTERN = (
    r"(?<![0-9])"
    r"(?:00[1-9]|0[1-9][0-9]|[1-578][0-9]{2}|6[0-57-9][0-9]|66[0-57-9])"
    r"(?P<ssn_sep>[- ])"
    r"(?:0[1-9]|[1-9][0-9])"
    r"(?P=ssn_sep)"
    r"(?:000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})"
)

_EMAIL_RE = re.compile(EMAIL_PATTERN)
_CARD_RE = re.compile(CARD_PATTERN)
_SSN_RE = re.compile(SSN_PATTERN)
# Prefilter: the leftmost position where *some* detector regex matches. Validation happens after.
_CANDIDATE_RE = re.compile(f"{EMAIL_PATTERN}|{CARD_PATTERN}|{SSN_PATTERN}")
# Same, minus emails: used when no "@" is in reach, which is most of the time.
_NUMBER_CANDIDATE_RE = re.compile(f"{CARD_PATTERN}|{SSN_PATTERN}")

# Viable suffixes, matched against the *reversed* tail of the buffer (anchored at the buffer end).
#   email: local{0,64}   or   local{1,64} "@" domain-chars{0,DOMAIN_MAX}
#   digits: d ([ -]? d){0,18} [ -]?     (covers both card and SSN prefixes)
# They over-approximate what the detector regexes can consume, which is what soundness needs.
_EMAIL_TAIL_REV = re.compile(
    rf"[\w.-]{{0,{DOMAIN_MAX}}}+@{_LOCAL}{{1,{LOCAL_MAX}}}+|{_LOCAL}{{0,{LOCAL_MAX}}}+"
)
_DIGIT_TAIL_REV = re.compile(rf"[ -]?(?:[0-9][ -]?){{0,{CARD_MAX_DIGITS - 1}}}[0-9]")

_EMAIL_VIABLE_MAX = LOCAL_MAX + 1 + DOMAIN_MAX
_DIGIT_VIABLE_MAX = 2 * CARD_MAX_DIGITS
_TAIL_WINDOW = max(_EMAIL_VIABLE_MAX, _DIGIT_VIABLE_MAX)
_CONTEXT = 1  # characters kept before the decided position, for the (?<![0-9]) lookbehinds

MAX_BUFFERED = _TAIL_WINDOW + _CONTEXT
"""Upper bound on ``StreamRedactor.buffered`` after ``feed()`` returns."""


# ---------------------------------------------------------------------------------------------
# Validators: (text, match) -> end of the span to redact, or None to reject the candidate.
# They may look at text[match.end()], never further.
# ---------------------------------------------------------------------------------------------
def _valid_tld(label: str) -> bool:
    return (len(label) >= 2 and label.isalpha()) or (label.startswith("xn--") and len(label) > 4)


def _email_end(text: str, m: re.Match[str]) -> int | None:
    start, end = m.span()
    candidate = m.group()
    at = candidate.index("@")
    labels = candidate[at + 1 :].split(".")
    if end < len(text) and (text[end].isalnum() or text[end] in "_-"):
        labels.pop()  # the last label was cut at LABEL_MAX characters, so it is not a real label
    while len(labels) >= 2 and not _valid_tld(labels[-1]):
        labels.pop()  # "john@example.com.123" -> "john@example.com"
    if len(labels) < 2:
        return None
    return start + at + 1 + sum(map(len, labels)) + len(labels) - 1


_LUHN_DOUBLED = str.maketrans("0123456789", "0246813579")  # 2*d, minus 9 when above 9
_SEPARATOR_RE = re.compile(r"[ -]")


def _luhn_ok(digits: str) -> bool:
    doubled = digits[-2::-2].translate(_LUHN_DOUBLED)
    return (sum(map(int, digits[-1::-2])) + sum(map(int, doubled))) % 10 == 0


def _card_end(text: str, m: re.Match[str]) -> int | None:
    start, end = m.span()
    raw = m.group()
    digits = raw.replace(" ", "").replace("-", "")
    # Places the card may end: before a separator, or at the end of the match unless the
    # 19-digit cap cut a longer digit group. Stored as (digits so far, end index), shortest first.
    cuts = [(i - j, start + i) for j, i in enumerate(sep.start() for sep in _SEPARATOR_RE.finditer(raw))]
    if end == len(text) or text[end] not in _DIGITS:
        cuts.append((len(digits), end))
    # Longest Luhn-valid prefix that ends on a digit-group boundary, so "4111 1111 1111 1111 2"
    # still redacts the card and a 20-digit order number is not cut in the middle of a group.
    for count, stop in reversed(cuts):
        if count < CARD_MIN_DIGITS:
            break
        if _luhn_ok(digits[:count]):
            return stop
    return None


def _ssn_end(text: str, m: re.Match[str]) -> int | None:
    end = m.end()
    if end < len(text) and text[end] in _DIGITS:
        return None  # part of a longer number
    return end


_Validator = Callable[[str, re.Match[str]], int | None]
_DETECTORS: tuple[tuple[re.Pattern[str], _Validator], ...] = (
    (_EMAIL_RE, _email_end),
    (_CARD_RE, _card_end),
    (_SSN_RE, _ssn_end),
)


# ---------------------------------------------------------------------------------------------
# The shared decision procedure.
# ---------------------------------------------------------------------------------------------
def _decide(text: str, start: int) -> int | None:
    """Return the end of the PII span starting at ``start``, or None if there is none."""
    for pattern, validate in _DETECTORS:
        m = pattern.match(text, start)
        if m is not None:
            end = validate(text, m)
            if end is not None:
                return end
    return None


def _redact_span(text: str, pos: int, limit: int, out: list[str], replacement: str) -> tuple[int, int]:
    """Decide every candidate starting in ``[pos, limit)`` and append the result to ``out``.

    Returns ``(new_pos, redactions)``. ``new_pos >= limit``. It can be larger when an accepted
    match runs past ``limit``.
    """
    count = 0
    search = pos
    # Cached lookups, refreshed only when `search` moves past them, so a text with many rejected
    # candidates is still scanned in linear time.
    at = text.find("@", search)
    number = _NUMBER_CANDIDATE_RE.search(text, search)
    while True:
        if 0 <= at < search:
            at = text.find("@", search)
        if number is not None and number.start() < search:
            number = _NUMBER_CANDIDATE_RE.search(text, search)
        # An email needs its "@" within LOCAL_MAX characters, so before that only numbers match.
        if at < 0 or (number is not None and number.start() < at - LOCAL_MAX):
            m = number
        else:
            m = _CANDIDATE_RE.search(text, max(search, at - LOCAL_MAX))
        if m is None or m.start() >= limit:
            break
        start = m.start()
        end = _decide(text, start)
        if end is None:
            search = start + 1
            continue
        if start > pos:
            out.append(text[pos:start])
        out.append(replacement)
        count += 1
        pos = search = end
    if limit > pos:
        out.append(text[pos:limit])
        pos = limit
    return pos, count


def _hold_point(text: str, pos: int) -> int:
    """Leftmost index >= ``pos`` whose suffix could still become (part of) a match."""
    n = len(text)
    tail = text[max(pos, n - _TAIL_WINDOW) : n][::-1]
    keep = 0
    m = _EMAIL_TAIL_REV.match(tail)
    if m is not None:
        keep = m.end()
    m = _DIGIT_TAIL_REV.match(tail)
    if m is not None and m.end() > keep:
        keep = m.end()
    return n - keep


def redact(text: str, replacement: str = REDACTED) -> str:
    """Redact a complete text in one pass."""
    out: list[str] = []
    _redact_span(text, 0, len(text), out, replacement)
    return "".join(out)


class StreamRedactor:
    """Redacts a text stream chunk by chunk with minimal hold-back.

    ``feed()`` returns the text that is final now. ``flush()`` must be called once at the end of the
    stream to release the held tail. One instance handles one text stream, so use one per
    ``choices[i]`` field of an LLM response.
    """

    __slots__ = ("_buf", "_pos", "redactions", "replacement")

    def __init__(self, replacement: str = REDACTED) -> None:
        self.replacement = replacement
        self.redactions = 0
        self._buf = ""  # _CONTEXT chars of already-emitted text, then the held tail
        self._pos = 0  # index in _buf where the held (undecided) tail starts

    @property
    def pending(self) -> str:
        """Text received but not emitted yet (the held tail)."""
        return self._buf[self._pos :]

    @property
    def buffered(self) -> int:
        """Characters currently kept in memory, including the lookbehind context."""
        return len(self._buf)

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        buf = self._buf + chunk
        out: list[str] = []
        pos, count = _redact_span(buf, self._pos, _hold_point(buf, self._pos), out, self.replacement)
        self.redactions += count
        keep_from = max(0, pos - _CONTEXT)
        self._buf = buf[keep_from:]
        self._pos = pos - keep_from
        return "".join(out)

    def flush(self) -> str:
        out: list[str] = []
        _, count = _redact_span(self._buf, self._pos, len(self._buf), out, self.replacement)
        self.redactions += count
        self._buf = ""
        self._pos = 0
        return "".join(out)
