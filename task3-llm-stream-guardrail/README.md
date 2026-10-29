# Task 3 — LLM streaming guardrail (PII redaction)

An incremental redaction engine that removes emails, payment card numbers and US SSNs from LLM
output **while it streams**. It holds back only the few characters that could still turn into a
match.

## Install and test

```bash
pip install -e ".[dev]"
pytest                                   # derandomized Hypothesis profile, about 2 s
HYPOTHESIS_PROFILE=thorough pytest       # 20,000 random examples per property
```

```python
from stream_guardrail import StreamRedactor, redact

r = StreamRedactor()
r.feed("Contact john.doe@ex")  # "Contact "; "john.doe@ex" is held back
r.feed("ample.com today")  # "[REDACTED] "; "today" could still become today@...
r.flush()  # "today"
redact("SSN 123-45-6789")  # "SSN [REDACTED]"
```

## What is detected

| Type | Rule | False-positive control |
|------|------|------------------------|
| Email | `local@label.label…`: local part ≤ 64 chars of `[\w.%+-]`, 2–8 labels of ≤ 63 chars of `[\w-]`, Unicode letters allowed | Last label must be a TLD (≥ 2 letters, or `xn--` punycode); invalid trailing labels are trimmed (`john@example.com.123` keeps `.123`) |
| Payment card | 13–19 digits, a single space or dash allowed between any two digits (`4111 1111 1111 1111`, `3782-822463-10005`) | **Luhn checksum**; not glued to other digits; must end on a digit-group boundary |
| US SSN | `AAA-GG-SSSS` or `AAA GG SSSS`, consistent separator | Never-issued numbers rejected: area `000`/`666`/`9xx`, group `00`, serial `0000`; not glued to other digits |

Every match is replaced by `[REDACTED]`. The replacement string is configurable.

### Precise semantics (one shot)

`redact(text)` scans left to right. At each position it tries **email, then card, then SSN**. The
first detector that matches *and* passes its validator wins: the span is replaced and scanning
resumes after it. If none applies, scanning moves one character right. Two consequences:

- An email that contains digits (`4111111111111111@example.com`) is redacted as one email.
- In a long run of digit groups the leftmost group start wins, and from there the **longest
  Luhn-valid run that ends on a group boundary**. So `4111 1111 1111 1111 2 times` becomes
  `[REDACTED] 2 times` instead of leaking the card because the trailing `2` broke the checksum.

## How streaming works

`StreamRedactor` keeps a rolling buffer: one character of left context (for the
`(?<![0-9])` lookbehinds) and the **held tail**, meaning text received but not yet final.
`feed(chunk)` does:

1. **Hold point.** Find the leftmost position whose suffix is still a *viable prefix* of some
   detector, e.g. `john.doe@ex`, `4111 1111` or the last word `today`. This is two anchored regex
   matches on the reversed tail of the buffer (`_EMAIL_TAIL_REV`, `_DIGIT_TAIL_REV`), so the cost
   is proportional to the trailing run, not to the buffer.
2. **Decide everything before it** with the same `_redact_span()` and `_decide()` code that
   `redact()` uses, then emit the result.
3. Keep the rest.

`flush()` decides the held tail as end-of-text and empties the buffer.

### Why decisions before the hold point are final

The detector regexes only consume characters from small alphabets, every quantifier is bounded
and **possessive** (no backtracking), and validators look at most one character past a match. If
any detector, starting at position `s`, read up to the end of the buffer, the text from `s` would
itself be a viable prefix, so `s` would be at or after the hold point. So every decision before the
hold point uses only characters that are already in the buffer, and it is the decision one-shot
redaction makes on the full text. This gives two invariants:

| Invariant | Test |
|-----------|------|
| **Chunking invariance**: for *any* split of a text, `feed()…+flush()` == `redact(text)` | Hypothesis with random texts built from PII fragments, random cut points, empty and 1-char chunks; exhaustive 2- and 3-way splits of tricky strings |
| **Bounded memory**: after `feed()`, `buffered <= MAX_BUFFERED` (577 chars) | Checked after every chunk in the property test and on a 5,000-chunk stream |
| **Nothing held across a newline** | Property test: after a chunk ending in `\n`, `pending == ""` |

### Why not `regex` partial matching?

The third-party `regex` module has `partial=True`. It does not answer the question streaming
needs, which is "can more input still change the result here?":

- it prefers a complete match over an earlier partial one: `a.{5}|b` searched in `"a12b"` returns
  `b`, although `a…` at position 0 may still match;
- greedy backtracking hides extensions: `[a-z]+@[a-z]+(\.[a-z]+)+` on `a@b.c.` reports a complete
  `a@b.c`, but one more letter makes it `a@b.c.d`;
- a lookahead at the end of the buffer is evaluated as if the text had ended.

Each of these breaks chunking invariance. Computing the hold point from the detector alphabets is
sound by construction, needs only the standard library, and is exact in the common cases: the last
word, a partial email, or a partial digit group.

## Complexity and memory

| | Bound |
|-|-------|
| Held tail | ≤ 64 chars for a run of word characters without `@` (the longest possible local part), ≤ 576 for `local@domain`, ≤ 38 for digit groups. Prose: one word. |
| Memory per stream | O(`MAX_BUFFERED`) = 577 chars, independent of response length |
| Work per `feed()` | O(len(chunk) + held tail). Candidates are found by C-level regex scans. `@` and number lookups are cached, so even one-shot redaction of a large text with many rejected candidates stays linear. |

## Known limitations

- **Obfuscated PII** (`john at example dot com`, digits spelled out, homoglyphs, full-width digits)
  is not detected. Only ASCII digits count for cards and SSNs.
- **Bare 9-digit SSNs** (`123456789`) are not redacted. Without separators they cannot be told
  apart from order numbers, ZIP+4 codes and so on.
- Emails with more than 8 host labels or a local part longer than 64 characters are outside the
  grammar. For the latter, only the last 64 characters before `@` and the domain are redacted.
- Card detection is checksum-based, so roughly 1 in 10 random 13–19 digit numbers that happen to
  pass Luhn are redacted too.
