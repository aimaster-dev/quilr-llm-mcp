# Task 3 — LLM streaming guardrail (PII redaction)

An OpenAI-compatible gateway for `POST /v1/chat/completions`. It proxies requests to an upstream
LLM provider and streams the response back with emails, payment card numbers and US SSNs
replaced by `[REDACTED]` **in flight**. A pattern split across chunks (`"john.doe@ex"` +
`"ample.com"`) is still caught. The gateway never buffers the response: it holds back only the few
characters that could still turn into a match.

## Run

```bash
pip install -e ".[dev]"
stream-guardrail-mock --port 9000 --chunk-size 8 --delay-ms 30  # terminal 1: fake provider
stream-guardrail --port 8080                                   # terminal 2: gateway -> :9000/v1
```

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mock-llm-1","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

```text
data: {"id":"chatcmpl-mock-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {…"choices":[{"index":0,"delta":{"content":"Sure! "},"finish_reason":null}]}

data: {…"choices":[{"index":0,"delta":{"content":"Here is "},"finish_reason":null}]}
…
data: {…"choices":[{"index":0,"delta":{"content":"[REDACTED]. SSN: "},"finish_reason":null}]}
…
data: [DONE]
```

Put together, the deltas read: *Sure! Here is the customer record you asked for. Name: Jane Roe.
Email: [REDACTED]. Card on file: [REDACTED], backup card [REDACTED]. SSN: [REDACTED]. Order number
1234567890123 and version 2.4.1 are not sensitive. Anything else?* (The order number fails Luhn,
so it is not a card.) Add `"mock": {"echo": true}` to the body to have your own message streamed
back through the filter.

The mock provider can be tuned with CLI flags or, per request, with a `"mock"` object in the body
(the gateway forwards it untouched): `chunk_size`, `delay_ms`, `first_token_delay_ms`
(`--ttft-ms`), `repeat`, `text` and `echo`. It supports `n` and `stream_options.include_usage`.
Setting `MOCK_LLM_API_KEY` makes it require that bearer token.

Against a real provider:
`GUARDRAIL_UPSTREAM_API_KEY=sk-... stream-guardrail --upstream-base-url https://api.openai.com/v1`.

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
- **Deny by default.** Every string under a choice's `delta` (or `message`) is treated as model
  output and redacted, whatever the field: `content` (a string or an array of content parts),
  `refusal`, `reasoning_content`, `reasoning`, `audio.transcript`, tool-call and legacy
  `function_call` arguments, and fields the gateway has never heard of. Only structural values are
  exempt: `role`, `type`, `id`, `index` and a tool's `function.name`. Other choice fields (say a
  legacy `text`) are redacted one shot.
- **Token logprobs are refused.** `logprobs` repeats every generated token with its bytes, which
  cannot be redacted token by token. Requests with `logprobs` or `top_logprobs` get a 400, and a
  `logprobs` object the upstream sends anyway is replaced by `null`. Audio output (`modalities`
  with `audio`, or an `audio` parameter) is refused for the same reason: speech cannot be redacted.
- **One redactor per text stream.** Each field is its own stream, keyed by `(choices[i].index,
  field path)`. Tool calls are identified by their `index` field and content parts by their
  `type`, not by array position, so `n > 1` and parallel tool calls never share state. A text
  content part continues the `content` string, as clients read both as the message text.
- **Flush points.** When a choice's `finish_reason` chunk arrives, the text that choice still holds
  in any field goes out in a synthetic chunk **just before** it (same `id`/`model`/`created`), and
  the finish chunk keeps an empty delta. The same happens before `data: [DONE]` and at EOF, so
  clients that stop reading at either point miss nothing.
- **No extra latency.** Each event is written to the client as soon as it is redacted, even when
  one upstream read carried many. Chunks whose text is entirely held back, and that carry nothing
  else (no role, id, `finish_reason`, `usage` or other choice data), are dropped instead of being
  sent empty. Comments (`: keep-alive`) and other non-content events pass through.
- **Non-streaming** (`"stream": false`): the same deny-by-default rules apply to
  `choices[].message`, one shot.
- **Payloads it does not recognise** (a mid-stream `{"error": …}`, non-JSON `data:`) are redacted
  as plain text rather than passed through.
- **Client disconnect cancels the upstream request** immediately, in every phase: while the
  provider is still thinking (no status line sent yet), while a non-streaming answer is being
  read, and while a stream is relayed. The endpoint is a plain ASGI app: once the request body is
  read, one task watches `receive()` for `http.disconnect` for the rest of the request and cancels
  the work, and the upstream response is closed in a shielded `finally`. (Starlette's
  `StreamingResponse` only watches while streaming, and only on ASGI spec < 2.4 servers.)
- **Nothing unexpected reaches the client.** JSON goes out with ASCII escapes, so a lone surrogate
  from the provider (`"\ud83d"`) cannot break the encoding. A bug while relaying a stream ends it
  with held text flushed and an in-band `internal_error` event, not a broken chunked body; anything
  else becomes a 500 in the OpenAI error shape, with details only in the log.
- **Credentials.** The client's `Authorization` and cookies are never forwarded. Only the
  gateway's own key, `x-request-id` (if it is a safe token) and `accept-encoding: identity` go
  upstream. `identity` matters because compression would make the provider buffer tokens.

### Errors

Upstream bodies, exception text and internal addresses never reach the client. Errors use
OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`.

| Situation | Client gets |
|-----------|-------------|
| Body is not JSON (or nested too deeply to parse), or not an object / too large | 400 `invalid_json` / `invalid_request`, 413 `request_too_large` (upstream not called) |
| `logprobs` / `top_logprobs` requested | 400 `logprobs_not_supported`, `param` names the field (upstream not called) |
| Audio output requested (`modalities`, `audio`) | 400 `audio_not_supported` (upstream not called) |
| Upstream 400/404/409/413/415/422 | Same status, `upstream_rejected_request`, plus `upstream_code` if the provider sent a safe machine-readable code such as `model_not_found` |
| Upstream 401/403 (the gateway's key) | 502 `upstream_auth_failed` |
| Upstream 429 | 429 `rate_limited`, `Retry-After` passed through if it is a plain token |
| Upstream 408/504, connect/read timeout | 504 `upstream_timeout` |
| Upstream 5xx, connection refused | 502 `upstream_error` / `upstream_unavailable` |
| Upstream 3xx (redirects are never followed) or any other non-2xx | 502 `upstream_error`; no `Location`, no body |
| Unexpected gateway error before the response started | 500 `internal_error` (`type: server_error`) |
| Failure **after** streaming started | Held text is flushed (redacted), then `data: {"error": {…"code":"upstream_stream_error"}}`; no `[DONE]` |
| Unexpected gateway error **after** streaming started | Held text is flushed (if the redactor can), then an in-band `internal_error` event; no `[DONE]` |

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
| Email | `local@label.label…`: local part ≤ 64 chars of word characters, `.%+-` and combining marks; 2–8 labels of ≤ 63 chars of word characters and `-`. Unicode letters are allowed (`jürgen@exämple.de`, NFC or NFD) except CJK: Han, Kana and Hangul end the address, so `请联系john@example.com获取` becomes `请联系[REDACTED]获取` | Last label must be a TLD: ≥ 2 letters and not a file extension (`icon@2x.png`), or `xn--` punycode. Invalid trailing labels are trimmed (`john@example.com.123` keeps `.123`). A git SSH remote (`git@github.com:org/repo.git`) is kept as is |
| Payment card | 13–19 digits starting with 2–6 (Visa, Mastercard, Amex, Discover, JCB, Diners, UnionPay…), either contiguous or grouped as cards are printed: first group of 4 digits, every group but the last ≥ 4 (4-4-4-4, 4-4-4-4-3, Amex 4-6-5, Diners 4-6-4), one separator between groups, all dash-like or all space-like | **Luhn checksum**; not glued to other digits; the longest valid prefix that ends on a group boundary. So `2 3 5 7 11 13…`, dates and a phone number before a card are not cards |
| US SSN | `AAA-GG-SSSS` or `AAA GG SSSS`: both separators dash-like or both space-like (`123-45 6789` is not an SSN) | Never-issued numbers rejected: area `000`/`666`/`9xx`, group `00`, serial `0000`; not glued to other digits |

Separators are one shared class: ASCII space and hyphen-minus, plus what LLMs and word processors
emit: U+2010 hyphen, **U+2011 non-breaking hyphen**, figure, en and em dashes, U+2212 minus, small
and fullwidth hyphen-minus; NBSP, the U+2000–U+200A spaces (figure, thin…), U+202F narrow NBSP,
U+205F and U+3000.

Every match is replaced by `[REDACTED]`. The replacement string is configurable.

### Precise semantics (one shot)

`redact(text)` scans left to right. At each position it tries **email, then card, then SSN**. The
first detector that matches *and* passes its validator wins: the span is replaced and scanning
resumes after it. If none applies, scanning moves one character right. Two consequences:

- An email that contains digits (`4111111111111111@example.com`) is redacted as one email.
- In a run of digit groups the leftmost group start that begins a card wins, and from there the
  **longest Luhn-valid run that ends on a group boundary**. So `4111 1111 1111 1111 2 times`
  becomes `[REDACTED] 2 times` instead of leaking the card because the trailing `2` broke the
  checksum, and `call 555 0100 4111 1111 1111 1111` becomes `call 555 0100 [REDACTED]`.
- A git remote is skipped as a whole, so `it@github.com` inside `git@github.com:…` is not
  redacted either.

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
and **possessive** (no backtracking), there are no lookaheads, and validators look at most one
character past a match. If any detector, starting at position `s`, read up to the end of the
buffer, it consumed everything from `s`, so that text would itself be a viable prefix and `s`
would be at or after the hold point. So every decision before the hold point uses only characters
that are already in the buffer, and it is the decision one-shot redaction makes on the full text.
The tail patterns may over-approximate (accept more than a detector can consume): that only holds
text a little longer. The email tail uses plain `\w` classes for speed and the held part is then
cut at the last CJK character, which no detector consumes. This gives two invariants:

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
| Held tail | ≤ 64 chars for a run of word characters without `@` (the longest possible local part), ≤ 576 for `local@domain` (64 + 1 + 8 labels of 63 + 7 dots), ≤ 38 for digit groups (19 digits, 19 one-character separators). Prose: one word. CJK prose: nothing. |
| Memory per stream | O(`MAX_BUFFERED`) = 577 chars per choice/field, plus one partial SSE line, independent of response length |
| Work per `feed()` | O(len(chunk) + held tail). Candidates are found by C-level regex scans. `@` and number lookups are cached, so even one-shot redaction of a large text with many rejected candidates stays linear. |

## Tests

`pytest` runs 129 tests in about 6 s, deterministic and with no fixed ports.
`HYPOTHESIS_PROFILE=thorough pytest` runs 20,000 random examples per property.

| Area | What is proven |
|------|----------------|
| Engine (`test_redaction*.py`) | Detector semantics; **streaming == one-shot for any chunking** (Hypothesis + exhaustive 2/3-way splits); buffer ≤ `MAX_BUFFERED` after every chunk; nothing held across `\n`; long runs released |
| SSE (`test_sse.py`) | Same events for every byte split point, byte-by-byte and random splits (UTF-8, `\r\n`, `\r`) |
| Chunk transformer (`test_completions.py`) | Every delta and message field redacted (content parts, reasoning, audio transcript, tool and legacy function arguments, unknown fields), structural values intact, logprobs dropped, tool calls keyed by `index`; Hypothesis: any interleaving of fields and choices reads as one-shot redaction, with nothing after `finish_reason`; `[DONE]`/EOF flush, dropped empty chunks |
| Gateway (`test_gateway.py`) | Redacted SSE, credential handling, refused logprobs/audio, sanitized error mapping (non-2xx, lone surrogates, deep JSON, unexpected exceptions in-band and as a 500), mid-stream failure, **disconnect cancels upstream** before the status line, while reading a complete answer and while streaming |
| Latency and memory (`test_streaming_latency.py`) | **First safe delta reaches the client while the upstream is blocked** waiting for it (a buffering gateway deadlocks); held text waits for one delta only; redactor peak < 64 chars over 5,000+ deltas; traced memory ~100 KB for a 1 MB stream |
| Real sockets (`test_end_to_end.py`) | Same guarantees over uvicorn + loopback TCP, including disconnect cancellation; a raw socket closed while the provider is still thinking (streaming or not) makes the gateway hang up on the provider |

`httpx.ASGITransport` buffers the whole response, so it cannot show streaming. The tests drive the
gateway through the raw ASGI interface (`tests/asgi.py`) and reach the mock provider through
`stream_guardrail.testing.StreamingASGITransport`, which hands over each body chunk as it is sent,
with backpressure.

## Benchmarks

```bash
python benchmarks/bench_redactor.py     # ~1 min; --quick for a smoke run
```

Measured on a busy Windows 11 laptop, Python 3.12. For section 3, the servers run in a child
process and every path is hit in interleaved order.

**1. Engine** (`StreamRedactor.feed()`, 200 KB of LLM-like text with PII about every KB):

| chunk size | µs per chunk | of which redaction | throughput |
|---:|---:|---:|---:|
| 1 char | 1.07 | 1.04 | 0.9 MB/s |
| 4 chars (≈ 1 token) | 1.22 | 1.19 | 3.3 MB/s |
| 64 chars | 4.31 | 4.28 | 14.9 MB/s |
| 4 KB | 207 | 207 | 19.7 MB/s |

**2. Gateway pipeline per upstream event** (4-char deltas): 3.0 µs to parse and re-encode the SSE
event, 9.3 µs with JSON and redaction, so **the guardrail adds about 6 µs of CPU per token**. A
model emitting 100 tokens/s costs under 0.1 % of a core.

**3. End to end over loopback TCP** (p50 / p95 ms, 200 unpaced and 20 paced requests per path):

| path | first byte | first content | total |
|---|---:|---:|---:|
| *mock answers instantly* | | | |
| direct to mock | 3.95 / 5.64 | 4.00 / 5.89 | 6.69 / 10.20 |
| gateway, redaction off | 7.16 / 11.94 | 11.07 / 17.66 | 13.92 / 21.62 |
| gateway, redaction on | 7.12 / 12.77 | 12.20 / 20.57 | 14.55 / 24.53 |
| *TTFT 50 ms, then a 4-char delta every 20 ms* | | | |
| direct to mock | 66.30 / 73.87 | 66.73 / 74.16 | 410.05 / 423.33 |
| gateway, redaction off | 71.27 / 75.65 | 71.51 / 76.32 | 417.18 / 426.28 |
| gateway, redaction on | 69.29 / 81.35 | 99.98 / 112.09 | 417.91 / 435.16 |

What this shows:

- **Time to first byte does not change** with redaction on. The role chunk is forwarded as soon as
  it arrives. The proxy hop itself (uvicorn + httpx on Windows loopback) costs about 3 ms.
- **Total stream time does not change** either: about 0.6 ms more over 65 deltas when the upstream
  is instant, and within noise at realistic pacing.
- **First content can arrive one delta later.** The mock's first delta is `Sure`, which could still
  become `Sure@example.com`, so it waits for `! He` (20 ms here, plus Windows timer granularity).
  This is the minimum price of never leaking a pattern split across chunks. A first delta that ends
  in a space or punctuation is forwarded at once (`test_first_safe_chunk…`).

## Known limitations

- **Request side is not filtered.** Only the model's output is redacted; prompts go upstream
  unchanged. Client authentication is also out of scope: run the gateway behind your API gateway.
- **Tool-call arguments are redacted as text.** `{"email": "a@b.co"}` stays valid JSON, but a card
  number written as a bare JSON number becomes `[REDACTED]` and makes the arguments invalid JSON.
  This is deliberate: breaking the call beats leaking the number. JSON-escaped PII
  (`a@b.co`) is not decoded first.
- **Unknown fields are treated as text streams.** A string field the gateway does not know (an
  annotation URL, say) may have its last word held back and delivered in a later chunk, which is
  harmless for clients that concatenate deltas. Text split across two *different* fields is not
  joined. Redaction also shifts character offsets, so `annotations[].start_index`/`end_index`
  can point at the wrong text.
- **Obfuscated PII** (`john at example dot com`, digits spelled out, homoglyphs, full-width digits,
  zero-width characters, Markdown inside the number such as `4111 **1111** …`) is not detected.
  Only ASCII digits count for cards and SSNs, dots are not separators (`4111.1111.1111.1111`), and
  a separator is one character (`4111  1111` with two spaces is two numbers).
- **Separator kinds are not mixed.** `123-45 6789` or `4111-1111 1111-1111` is not matched; mixed
  dashes (`-` and U+2011) or mixed spaces (space and NBSP) are.
- **Bare 9-digit SSNs** (`123456789`) are not redacted. Without separators they cannot be told
  apart from order numbers, ZIP+4 codes and so on. A phone number written `212 55 1234` has the
  SSN shape and is redacted.
- **Cards outside IIN 2–6** (UATP `1…`, some RuPay `8…`) and unusual groupings (a first group
  other than 4 digits) are not detected. Among 2–6 numbers, roughly 1 in 10 random contiguous
  13–19 digit ids passes Luhn and is redacted too.
- **CJK addresses.** Because CJK text ends an address, an address written in CJK script
  (`用户@例子.中国`) is not detected; its punycode form is. Only `git@…` remotes are recognised as
  non-addresses; `deploy@host:/path` is still redacted.
- Emails with more than 8 host labels or a local part longer than 64 characters are outside the
  grammar. For the latter, only the last 64 characters before `@` and the domain are redacted.
