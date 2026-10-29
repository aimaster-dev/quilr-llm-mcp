# Task 3 — LLM streaming guardrail (PII redaction)

An OpenAI-compatible gateway for `POST /v1/chat/completions`. It proxies requests to an upstream
LLM provider and streams the response back with emails, payment card numbers and US SSNs
replaced by `[REDACTED]` **in flight**. A pattern split across chunks (`"john.doe@ex"` +
`"ample.com"`) is still caught. The gateway never buffers the response: it holds back only the few
characters that could still turn into a match.

## Run

```bash
pip install -e ".[dev]"
export GUARDRAIL_UPSTREAM_API_KEY=sk-...           # the gateway's own provider key (env only)
stream-guardrail --upstream-base-url https://api.openai.com/v1 --port 8080
```

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gpt-4o-mini","stream":true,
       "messages":[{"role":"user","content":"Invent a customer record with an email and SSN"}]}'
```

```text
data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"content":"Name: Jane Roe, email "},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"content":"[REDACTED], SSN "},"finish_reason":null}]}
…
data: [DONE]
```

Tests: `pytest` (derandomized Hypothesis profile, about 2 s) or `HYPOTHESIS_PROFILE=thorough pytest`
(20,000 random examples per property).

### Configuration

| Env var (`GUARDRAIL_…`) | CLI flag | Default | |
|---|---|---|---|
| `UPSTREAM_BASE_URL` | `--upstream-base-url` | `http://127.0.0.1:9000/v1` | Provider base URL; `/chat/completions` is appended |
| `UPSTREAM_API_KEY` | – | unset | Sent upstream as `Authorization: Bearer …`. Env only, so it never shows up in `ps` |
| `CONNECT_TIMEOUT` | `--connect-timeout` | 5 s | |
| `READ_TIMEOUT` | `--read-timeout` | 60 s | Longest silence allowed between two upstream reads |
| `MAX_REQUEST_BYTES` | – | 1 MiB | Larger bodies get 413 |
| `MAX_RESPONSE_BYTES` | – | 8 MiB | Non-streaming responses only |
| `MAX_EVENT_SIZE` | – | 1 Mi chars | One SSE event; protects against a stream with no line breaks |
| `LOG_LEVEL` | `--log-level` | `INFO` | Logs include status, redaction count and timings, never content |

## Gateway behaviour

```text
upstream bytes ─▶ SSEParser ─▶ ChatStreamRedactor ─▶ SSE bytes ─▶ client
                  (partial line)  (one StreamRedactor per choice/field)
```

- **Incremental SSE parsing.** Upstream bytes can split anywhere: inside a multi-byte UTF-8
  character (incremental decoder), between `\r` and `\n`, or mid-field. The parser keeps one
  partial line and yields each event as soon as its blank line arrives.
- **One redactor per text stream.** Redactors are keyed by `choices[i].index` for `content` and
  `refusal`, and by `(choice, tool_call.index)` for `tool_calls[].function.arguments`, so `n > 1`
  and parallel tool calls never share state.
- **Flush points.** When a choice's `finish_reason` chunk arrives, the text that choice still holds
  goes out in a synthetic chunk **just before** it (same `id`/`model`/`created`), and the finish
  chunk keeps an empty delta. The same happens before `data: [DONE]` and at EOF, so clients that
  stop reading at either point miss nothing.
- **No extra latency.** Everything one upstream read produces is written to the client before the
  next read. Chunks whose text is entirely held back, and that carry nothing else (no role,
  `finish_reason`, `usage` or logprobs), are dropped instead of being sent empty. Comments
  (`: keep-alive`) and other non-content events pass through.
- **Non-streaming** (`"stream": false`): `choices[].message.content`, `refusal` and tool-call
  arguments are redacted in the JSON response.
- **Payloads it does not recognise** (a mid-stream `{"error": …}`, non-JSON `data:`) are redacted
  as plain text rather than passed through.
- **Client disconnect cancels the upstream request** immediately, even while the model is silent.
  Starlette's `StreamingResponse` only watches for disconnects on ASGI spec < 2.4 servers.
  `EventStreamResponse` always watches `receive()`, and the relay closes the upstream response in
  a shielded `finally`.
- **Credentials.** The client's `Authorization` and cookies are never forwarded. Only the
  gateway's own key, `x-request-id` (if it is a safe token) and `accept-encoding: identity` go
  upstream. `identity` matters because compression would make the provider buffer tokens.

### Errors

Upstream bodies, exception text and internal addresses never reach the client. Errors use
OpenAI's shape: `{"error": {"message", "type", "code"}}`.

| Situation | Client gets |
|-----------|-------------|
| Body is not JSON, or not an object / too large | 400 `invalid_json` / `invalid_request`, 413 `request_too_large` (upstream not called) |
| Upstream 400/404/409/413/415/422 | Same status, `upstream_rejected_request`, plus `upstream_code` if the provider sent a safe machine-readable code such as `model_not_found` |
| Upstream 401/403 (the gateway's key) | 502 `upstream_auth_failed` |
| Upstream 429 | 429 `rate_limited`, `Retry-After` passed through if it is a plain token |
| Upstream 408/504, connect/read timeout | 504 `upstream_timeout` |
| Upstream 5xx, connection refused | 502 `upstream_error` / `upstream_unavailable` |
| Failure **after** streaming started | Held text is flushed (redacted), then `data: {"error": {…"code":"upstream_stream_error"}}`; no `[DONE]` |

## Redaction engine

```python
from stream_guardrail import StreamRedactor, redact

r = StreamRedactor()
r.feed("Contact john.doe@ex")  # "Contact "; "john.doe@ex" is held back
r.feed("ample.com today")  # "[REDACTED] "; "today" could still become today@...
r.flush()  # "today"
redact("SSN 123-45-6789")  # "SSN [REDACTED]"
```

### What is detected

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

### How streaming works

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

#### Why decisions before the hold point are final

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

#### Why not `regex` partial matching?

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

### Complexity and memory

| | Bound |
|-|-------|
| Held tail | ≤ 64 chars for a run of word characters without `@` (the longest possible local part), ≤ 576 for `local@domain`, ≤ 38 for digit groups. Prose: one word. |
| Memory per stream | O(`MAX_BUFFERED`) = 577 chars per choice/field, plus one partial SSE line, independent of response length |
| Work per `feed()` | O(len(chunk) + held tail). Candidates are found by C-level regex scans. `@` and number lookups are cached, so even one-shot redaction of a large text with many rejected candidates stays linear. |

## Known limitations

- **Request side is not filtered.** Only the model's output is redacted; prompts go upstream
  unchanged. Client authentication is also out of scope: run the gateway behind your API gateway.
- **Tool-call arguments are redacted as text.** `{"email": "a@b.co"}` stays valid JSON, but a card
  number written as a bare JSON number becomes `[REDACTED]` and makes the arguments invalid JSON.
  This is deliberate: breaking the call beats leaking the number. JSON-escaped PII
  (`a@b.co`) is not decoded first. The legacy `function_call` field is not covered.
- **Obfuscated PII** (`john at example dot com`, digits spelled out, homoglyphs, full-width digits)
  is not detected. Only ASCII digits count for cards and SSNs.
- **Bare 9-digit SSNs** (`123456789`) are not redacted. Without separators they cannot be told
  apart from order numbers, ZIP+4 codes and so on.
- Emails with more than 8 host labels or a local part longer than 64 characters are outside the
  grammar. For the latter, only the last 64 characters before `@` and the domain are redacted.
- Card detection is checksum-based, so roughly 1 in 10 random 13–19 digit numbers that happen to
  pass Luhn are redacted too.
