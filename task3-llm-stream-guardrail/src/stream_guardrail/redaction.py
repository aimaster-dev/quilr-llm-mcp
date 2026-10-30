"""Incremental PII redaction for text that arrives in arbitrary chunks.

Semantics (one shot)
--------------------
``redact(text)`` scans left to right. At each position it tries the detectors in priority order
(email, payment card, US SSN). The first one that matches *and* passes its validator is replaced by
``[REDACTED]`` and scanning resumes after it. If none applies, scanning moves one character right.
One rule runs before them and works the other way: a git SSH remote (``git@host:path``) is not an
address, so it is kept as is and scanning resumes after it.

Streaming
---------
``StreamRedactor`` runs exactly the same procedure (``_redact_span`` and ``_decide`` below) over a
small rolling buffer. The only extra step is the *hold point*: the leftmost position whose suffix
could still grow into a match. A position before the hold point can be decided now, because no
detector can read past the end of the buffer from there:

* each detector regex only consumes characters from a small alphabet, and every quantifier is
  bounded. ``_EMAIL_TAIL_REV`` and ``_DIGIT_TAIL_REV`` accept every string a detector can consume
  (they may accept more): emails use word characters other than CJK, combining marks and
  ``.%+-@``; cards and SSNs use ASCII digits and the separator class ``_SEP``;
* the quantifiers are possessive, so the regexes never backtrack. They have no lookaheads; the
  ``(?<![0-9])`` lookbehinds read one character before the start, which the buffer keeps
  (``_CONTEXT``). The validators look at most one character past the match.

If a detector's search from ``s`` reached the end of the buffer, it consumed every character from
``s`` to the end, so that text would itself be a viable prefix, and ``s`` would be at or after the
hold point. So every decision made before the hold point is the decision one-shot redaction makes
on the full text. Text from the hold point onwards is kept back until more input arrives or the
stream ends. This gives the two invariants the tests check:

* **Chunking invariance**: for any split of a text into chunks, the concatenated output of
  ``feed()``/``flush()`` equals ``redact(text)``.
* **Bounded memory**: the buffer never holds more than ``MAX_BUFFERED`` characters after
  ``feed()`` returns, whatever the response length. The longest viable suffix is an email prefix
  of 64 + 1 + 511 characters (local part, ``@``, eight labels); a digit run is at most 19 digits
  and 19 single-character separators. One character of context is kept on top.

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

# Separators between digit groups: one shared class for cards, SSNs, the hold point and digit
# stripping. Besides ASCII space and hyphen-minus it has what LLMs and word processors emit:
# U+2010 hyphen, U+2011 non-breaking hyphen (common in model output), figure dash, en/em dash,
# U+2212 minus, small and fullwidth hyphens; NBSP, the U+2000-U+200A spaces (figure, thin...),
# U+202F narrow NBSP, U+205F and the ideographic space. Dash-like and space-like are kept apart:
# all separators of one SSN or one card must be of the same kind.
_DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
_SPACE_CHARS = " \u00a0" + "".join(map(chr, range(0x2000, 0x200B))) + "\u202f\u205f\u3000"
_DASHES = re.escape(_DASH_CHARS)  # for use inside [...]
_SPACES = re.escape(_SPACE_CHARS)
_SEP = f"[{_DASHES}{_SPACES}]"

# Characters a detector can consume. ``\w`` is Unicode-aware, so "jürgen@exämple.de" is covered,
# minus the CJK scripts (Han, Kana, Hangul): they are written without spaces, so in
# "请联系john@example.com获取帮助" the address must stop at the Han characters on both sides.
# Combining marks are included, so a decomposed (NFD) "josé" is one local part.
_CJK = (
    "\u1100-\u11ff\u3005-\u3007\u3040-\u30ff\u3130-\u318f\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    "\ua960-\ua97f\uac00-\ud7ff\uf900-\ufaff\uff66-\uffdc\U00020000-\U0003ffff"
)
_MARKS = "\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f"
_WORD = f"[^\\W{_CJK}]"
_LOCAL = f"(?:{_WORD}|[.%+\\-{_MARKS}])"
_LABEL_CHAR = f"(?:{_WORD}|[\\-{_MARKS}])"
_LABEL = rf"{_LABEL_CHAR}{{1,{LABEL_MAX}}}+"
_DIGITS = frozenset("0123456789")

EMAIL_PATTERN = rf"{_LOCAL}{{1,{LOCAL_MAX}}}+@{_LABEL}(?:\.{_LABEL}){{1,{MAX_LABELS - 1}}}+"
# 13-19 digits starting with 2-6 (the major networks' IINs), a single separator allowed between
# any two digits. The grouping rules (_card_end) decide which of these runs are cards.
CARD_PATTERN = rf"(?<![0-9])[2-6](?:{_SEP}?[0-9]){{{CARD_MIN_DIGITS - 1},{CARD_MAX_DIGITS - 1}}}+"
# AAA-GG-SSSS or AAA GG SSSS: both separators dash-like, or both space-like. Never-issued numbers
# are excluded: area 000, 666 and 900-999, group 00, serial 0000.
SSN_PATTERN = (
    r"(?<![0-9])"
    r"(?:00[1-9]|0[1-9][0-9]|[1-578][0-9]{2}|6[0-57-9][0-9]|66[0-57-9])"
    rf"(?:(?P<ssn_dash>[{_DASHES}])|[{_SPACES}])"
    r"(?:0[1-9]|[1-9][0-9])"
    rf"(?(ssn_dash)[{_DASHES}]|[{_SPACES}])"
    r"(?:000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})"
)

_EMAIL_RE = re.compile(EMAIL_PATTERN)
_AT_DOMAIN_RE = re.compile(rf"@[\w\-{_MARKS}]{{1,{LABEL_MAX}}}+\.")  # implied by EMAIL_PATTERN
_GIT_REMOTE_RE = re.compile(rf"git@{_LABEL}(?:\.{_LABEL}){{1,{MAX_LABELS - 1}}}+")
_CARD_RE = re.compile(CARD_PATTERN)
_SSN_RE = re.compile(SSN_PATTERN)
# Prefilter: the leftmost position where *some* detector regex matches. Validation happens after.
_CANDIDATE_RE = re.compile(f"{EMAIL_PATTERN}|{CARD_PATTERN}|{SSN_PATTERN}")
# Same, minus emails: used when no "@" is in reach, which is most of the time.
_NUMBER_CANDIDATE_RE = re.compile(f"{CARD_PATTERN}|{SSN_PATTERN}")

# Viable suffixes, matched against the *reversed* tail of the buffer (anchored at the buffer end).
#   email: local{0,64}   or   local{1,64} "@" domain-chars{0,DOMAIN_MAX}
#   digits: d (sep? d){0,18} sep?     (covers both card and SSN prefixes)
# They over-approximate what the detector regexes can consume, which is what soundness needs.
# The email classes are plain ``\w`` sets, which the regex engine repeats much faster than the
# CJK-excluding alternation; _hold_point then cuts the held text at the last CJK character, which
# no detector consumes. So CJK prose is never held back.
_EMAIL_TAIL_REV = re.compile(
    rf"[\w.\-{_MARKS}]{{0,{DOMAIN_MAX}}}+@[\w.%+\-{_MARKS}]{{1,{LOCAL_MAX}}}+"
    rf"|[\w.%+\-{_MARKS}]{{0,{LOCAL_MAX}}}+"
)
_DIGIT_TAIL_REV = re.compile(rf"{_SEP}?(?:[0-9]{_SEP}?){{0,{CARD_MAX_DIGITS - 1}}}[0-9]")
_CJK_RE = re.compile(f"[{_CJK}]")
# Last characters a digit-run suffix can end with (the reversed tail starts with them).
_DIGIT_TAIL_START = frozenset("0123456789" + _DASH_CHARS + _SPACE_CHARS)

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
# File extensions that are not TLDs, so "icon@2x.png" or "repo@v2.git" is a file name, not an
# address. Extensions that are real TLDs (.zip, .mov, .py, .md, .sh...) are not listed: an address
# there must still be redacted.
_FILE_EXTENSIONS = frozenset(
    {
        *("png", "jpg", "jpeg", "gif", "svg", "webp", "bmp", "ico", "tif", "tiff", "heic", "avif"),
        *("git", "js", "mjs", "cjs", "ts", "jsx", "tsx", "css", "scss", "json", "yaml", "yml"),
        *("toml", "xml", "html", "htm", "txt", "csv", "lock", "pdf", "doc", "docx", "xls", "xlsx"),
        *("ppt", "pptx", "gz", "tgz", "tar", "rar", "exe", "dll"),
    }
)
_LABEL_CHAR_RE = re.compile(_LABEL_CHAR)


def _valid_tld(label: str) -> bool:
    if label.startswith("xn--"):
        return len(label) > 4
    return len(label) >= 2 and label.isalpha() and label.lower() not in _FILE_EXTENSIONS


def _git_remote_end(text: str, m: re.Match[str]) -> int | None:
    # git@github.com:org/repo.git, ssh://git@host/repo: an SSH remote, not a mailbox. It is kept
    # as a whole, so the scanner does not go on to redact "it@github.com" inside it.
    end = m.end()
    return end if end < len(text) and text[end] in ":/" else None


def _email_end(text: str, m: re.Match[str]) -> int | None:
    start, end = m.span()
    candidate = m.group()
    at = candidate.index("@")
    labels = candidate[at + 1 :].split(".")
    if end < len(text) and _LABEL_CHAR_RE.match(text, end):
        labels.pop()  # the last label was cut at LABEL_MAX characters, so it is not a real label
    while len(labels) >= 2 and not _valid_tld(labels[-1]):
        labels.pop()  # "john@example.com.123" -> "john@example.com"
    if len(labels) < 2:
        return None
    return start + at + 1 + sum(map(len, labels)) + len(labels) - 1


_LUHN_DOUBLED = str.maketrans("0123456789", "0246813579")  # 2*d, minus 9 when above 9
_SEPARATOR_RE = re.compile(_SEP)
_DASH_RE = re.compile(f"[{_DASHES}]")


def _luhn_ok(digits: str) -> bool:
    doubled = digits[-2::-2].translate(_LUHN_DOUBLED)
    return (sum(map(int, digits[-1::-2])) + sum(map(int, doubled))) % 10 == 0


def _card_end(text: str, m: re.Match[str]) -> int | None:
    """Longest Luhn-valid card at the start of the run that ends on a digit-group boundary.

    A card is 13-19 contiguous digits, or digit groups as cards are printed: the first group has
    exactly 4 digits, every other group but the last has at least 4 (4-4-4-4, 4-4-4-4-3, Amex
    4-6-5, Diners 4-6-4), and the separators are all dash-like or all space-like. So a run of
    small numbers ("2 3 5 7 11 13...", dates, a phone number before a card) is not a card, and
    "4111 1111 1111 1111 2" still is, without the trailing group.
    """
    start, end = m.span()
    raw = m.group()
    first = _SEPARATOR_RE.search(raw)
    if first is not None and first.start() != 4 and first.start() < CARD_MIN_DIGITS:
        return None  # too short to be contiguous, and not a card grouping (fast path)
    groups = _SEPARATOR_RE.split(raw)
    separators = _SEPARATOR_RE.findall(raw)
    digits = "".join(groups)
    # Places the card may end, as (digits so far, end index), shortest first.
    count, stop = len(groups[0]), start + len(groups[0])
    cuts = [(count, stop)]  # contiguous digits
    if count == 4:
        dashed = _DASH_RE.match(separators[0]) is not None if separators else False
        for i in range(1, len(groups)):
            if (_DASH_RE.match(separators[i - 1]) is not None) != dashed:
                break  # mixed separator kinds
            if i >= 2 and len(groups[i - 1]) < 4:
                break  # a short group can only be the last one
            count += len(groups[i])
            stop += 1 + len(groups[i])
            cuts.append((count, stop))
    if cuts[-1][1] == end and end < len(text) and text[end] in _DIGITS:
        cuts.pop()  # the 19-digit cap cut a longer digit group: that is not where the number ends
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
# (pattern, validator, sensitive, needs an "@"). A span that is not sensitive is kept as is and
# skipped whole.
_DETECTORS: tuple[tuple[re.Pattern[str], _Validator, bool, bool], ...] = (
    (_GIT_REMOTE_RE, _git_remote_end, False, True),
    (_EMAIL_RE, _email_end, True, True),
    (_CARD_RE, _card_end, True, False),
    (_SSN_RE, _ssn_end, True, False),
)


# ---------------------------------------------------------------------------------------------
# The shared decision procedure.
# ---------------------------------------------------------------------------------------------
def _decide(text: str, start: int) -> tuple[int, bool] | None:
    """Return ``(end, sensitive)`` for the span starting at ``start``, or None if there is none."""
    # The address patterns need an "@" after 1-64 local-part characters. Without one they cannot
    # match, so skipping them changes nothing; it only saves two regex calls per digit run.
    has_at = text.find("@", start + 1, start + LOCAL_MAX + 1) >= 0
    for pattern, validate, sensitive, needs_at in _DETECTORS:
        if needs_at and not has_at:
            continue
        m = pattern.match(text, start)
        if m is not None:
            end = validate(text, m)
            if end is not None:
                return end, sensitive
    return None


def _next_at(text: str, start: int) -> int:
    """Index of the next "@" at or after ``start`` that can be part of an address, or -1.

    An address needs a label and a dot after its "@". Checking that with plain classes (a
    necessary condition) is much cheaper than trying every local-part start before an "@" that
    cannot work, such as a decorator or a handle. If the check runs into the end of the buffer,
    the text from the local part on is a viable prefix, so it is held and not decided yet.
    """
    at = text.find("@", start)
    while at >= 0 and _AT_DOMAIN_RE.match(text, at) is None:
        at = text.find("@", at + 1)
    return at


def _redact_span(text: str, pos: int, limit: int, out: list[str], replacement: str) -> tuple[int, int]:
    """Decide every candidate starting in ``[pos, limit)`` and append the result to ``out``.

    Returns ``(new_pos, redactions)``. ``new_pos >= limit``. It can be larger when an accepted
    match runs past ``limit``.
    """
    count = 0
    search = pos
    # Cached lookups, refreshed only when `search` moves past them, so a text with many rejected
    # candidates is still scanned in linear time.
    at = _next_at(text, search)
    number = _NUMBER_CANDIDATE_RE.search(text, search)
    while True:
        if 0 <= at < search:
            at = _next_at(text, search)
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
        decision = _decide(text, start)
        if decision is None:
            search = start + 1
            continue
        end, sensitive = decision
        if sensitive:
            if start > pos:
                out.append(text[pos:start])
            out.append(replacement)
            count += 1
        else:
            out.append(text[pos:end])  # kept as is; nothing inside it is a candidate
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
    if m is not None and m.end():
        keep = m.end()
        if not tail.isascii():  # O(1) for str
            cjk = _CJK_RE.search(tail, 0, keep)
            if cjk is not None:
                keep = cjk.start()  # a suffix that contains a CJK character is not viable
    if tail[:1] in _DIGIT_TAIL_START:
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
