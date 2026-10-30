# Task 4 — LLM gateway router (rate limiting + model failover)

An OpenAI-compatible `POST /v1/chat/completions` gateway that:

- limits each tenant API key to **50,000 tokens per rolling 60 s window**, with an exact
  sliding-window log stored in an **on-disk SQLite database** shared by all worker processes;
- gives the primary model a hard **3000 ms** deadline and fails over to a secondary provider on a
  primary **429** or **timeout** (and, configurably, on connection errors and 5xx);
- returns every error in one sanitized shape that never leaks upstream bodies, stack traces,
  hostnames or provider names.

Built on Starlette (ASGI), `httpx.AsyncClient`, anyio and the standard-library `sqlite3`.

## Run

```bash
pip install -e ".[dev]"

# Two mock providers (see "Mock provider" below)
mock-llm-provider --name primary   --port 9001
mock-llm-provider --name secondary --port 9002

# The gateway. Tenant keys and provider API keys come from the environment only.
export LLM_ROUTER_TENANT_KEYS=demo-key-1,demo-key-2   # PowerShell: $env:LLM_ROUTER_TENANT_KEYS="..."
llm-router --primary-url http://127.0.0.1:9001/v1 --primary-model primary-model \
           --secondary-url http://127.0.0.1:9002/v1 --secondary-model backup-model \
           --tokens-per-window 2000            # small budget so the demo hits the limit
```

### Walkthrough

The mock reports the whole `max_tokens` budget as used, so each of these requests costs 509 tokens
(9 for the prompt + 500).

```bash
REQ='{"model":"gpt-test","messages":[{"role":"user","content":"Hello"}],"max_tokens":500}'
chat() { curl -s -i http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $1" -d "$REQ"; }

chat demo-key-1
# HTTP/1.1 200 OK
# x-ratelimit-limit: 2000
# x-ratelimit-remaining: 1491
# x-gateway-route: primary
# x-request-id: req_9d636706ff5e496483078f2424a34af1
# {"id":"chatcmpl-...","object":"chat.completion","model":"gpt-test","choices":[{"index":0,
#  "message":{"role":"assistant","content":"Hello from the primary mock provider (request #1)."},
#  "finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":500,"total_tokens":509}}

# Primary starts answering 429 -> served by the secondary
curl -s -X PUT http://127.0.0.1:9001/_mock/mode -d '{"mode":"429"}'
chat demo-key-2            # 200, x-gateway-route: fallback, "Hello from the secondary mock provider ..."

# Primary hangs -> failover at the 3000 ms deadline (curl reports total=3.02s)
curl -s -X PUT http://127.0.0.1:9001/_mock/mode -d '{"mode":"slow","delay":10}'
chat demo-key-2            # 200, x-gateway-route: fallback

# ...and the secondary is broken too -> sanitized 503
curl -s -X PUT http://127.0.0.1:9002/_mock/mode -d '{"mode":"500"}'
chat demo-key-2
# HTTP/1.1 503 Service Unavailable
# {"error":{"type":"upstream_error","code":"upstream_unavailable","message":"No model provider is
#  available to serve this request right now. Please retry later.","request_id":"req_104d5542..."}}

# Back to healthy; exhaust demo-key-1's window
curl -s -X PUT http://127.0.0.1:9001/_mock/mode -d '{"mode":"ok"}'
curl -s -X PUT http://127.0.0.1:9002/_mock/mode -d '{"mode":"ok"}'
chat demo-key-1; chat demo-key-1; chat demo-key-1
# HTTP/1.1 429 Too Many Requests
# x-ratelimit-limit: 2000
# x-ratelimit-remaining: 473
# retry-after: 38
# {"error":{"type":"rate_limit_error","code":"tenant_token_limit_exceeded","message":"Token rate
#  limit exceeded for this API key. Retry after 38 seconds.","request_id":"req_6d4c2555..."}}
```

The mock's 500 body contains a fake traceback, an internal hostname and a fake API key. The client
saw none of it; the server log has all of it under the same request id:

```text
WARNING llm_router.router [req_104d...] primary attempt failed: kind=timeout status=None elapsed_ms=3016 detail=no complete response within 3000 ms
WARNING llm_router.router [req_104d...] fallback attempt failed: kind=server_error status=500 elapsed_ms=0 detail='Traceback (most recent call last):\n  File "/srv/secondary/...
WARNING llm_router.middleware [req_104d...] POST /v1/chat/completions -> 503 upstream_unavailable attempts=[primary:timeout, fallback:server_error(500)]
```

`llm-router --workers 4` runs four server processes; they share one budget through the SQLite file.
**Not on Windows:** uvicorn's multi-process mode there can intermittently leave concurrent
connections unanswered (plain uvicorn `--workers` shows the same thing), so `llm-router` logs a
warning when `--workers` is above 1 on Windows. Use one worker on Windows; scale out on Linux.

### Configuration

Every flag also reads an `LLM_ROUTER_*` environment variable (`--primary-timeout-ms` ->
`LLM_ROUTER_PRIMARY_TIMEOUT_MS`); flags win.

| Flag | Default | Notes |
|------|---------|-------|
| `--primary-url`, `--primary-model` | required | OpenAI-compatible base URL (`.../v1`) |
| `--primary-timeout-ms` | `3000` | hard deadline for the whole primary attempt |
| `--secondary-url`, `--secondary-model` | none | without them there is no failover |
| `--secondary-timeout-ms` | `30000` | deadline for the secondary attempt |
| `--[no-]failover-on-connection-error` | on | 429 and timeouts always fail over |
| `--[no-]failover-on-server-error` | on | 5xx and malformed or oversized 2xx bodies |
| `--[no-]failover-on-refused` | on | 4xx other than 400/413/422/429 (401, 403, 404, 408, ...) |
| `--max-response-bytes` | `4194304` | larger provider bodies are a bad response |
| `--tokens-per-window`, `--window-seconds` | `50000`, `60` | per tenant API key |
| `--db-path` | `data/llm_router.db` | must be on a local disk (SQLite locking) |
| `--db-busy-timeout-ms` | `1000` | how long a limiter transaction waits for the SQLite write lock |
| `--db-fail-fast-ms` | `1000` | after a limiter database error, answer 503 at once for this long (`0` disables) |
| `--sweep-interval` | `30` | seconds between global evictions |
| `--default-max-tokens` | `1024` | used, and sent upstream, when the client omits `max_tokens` |
| `--max-body-bytes` | `1048576` | larger bodies get `413 payload_too_large` |
| `--host`, `--port`, `--workers`, `--log-level` | `127.0.0.1`, `8080`, `1`, `INFO` | `--workers` above 1: Linux/macOS only (see above) |

| Environment only | |
|------|---|
| `LLM_ROUTER_TENANT_KEYS` | comma-separated tenant keys; hashed at startup, never stored |
| `LLM_ROUTER_TENANT_KEY_SHA256` | comma-separated SHA-256 fingerprints, so the gateway never needs the raw keys |
| `LLM_ROUTER_PRIMARY_API_KEY`, `LLM_ROUTER_SECONDARY_API_KEY` | sent upstream as `Authorization: Bearer` |

## API

`POST /v1/chat/completions` with `Authorization: Bearer <tenant-key>` and a strict subset of the
OpenAI body: `messages` (roles `system`/`developer`/`user`/`assistant`, string or text-part
content), `model`, `max_tokens` (1-32768), `temperature`, `top_p`, `stop`. Unknown fields are
rejected rather than forwarded. `stream: true` gets `400 streaming_not_supported`: failover is only
clean before the first byte reaches the client, and a non-streaming endpoint keeps that guarantee
simple. `model` is echoed back; the gateway decides which provider models serve the request.

| Response header | Meaning |
|-----------------|---------|
| `X-Request-Id` | on every response; also in error payloads and every log line |
| `X-RateLimit-Limit` / `X-RateLimit-Remaining` | the tenant's window budget and what is left after this request |
| `Retry-After` | on 429: whole seconds until enough of the window drains for this request; on 503: the smallest sanitized `Retry-After` the providers sent (1-60 s), if any |
| `X-Gateway-Route` | `primary` or `fallback`: an opaque label, not a provider name |

`GET /healthz` returns `{"status": "ok"}`.

## Rate limiter

### Algorithm: sliding-window log

Each admitted request inserts one row. A tenant's usage at time `now` is the sum of its rows with
`ts > now - 60`. That is exact: a fixed-window counter lets a tenant spend 50k at 0:59 and another
50k at 1:00, and a sliding *counter* only approximates. The cost is one row per request, cleaned up
continuously.

```sql
CREATE TABLE reservations (
    id      TEXT PRIMARY KEY,      -- random UUID, never reused (see below)
    tenant  TEXT NOT NULL,         -- SHA-256 fingerprint of the API key, never the key
    ts      REAL NOT NULL,         -- wall-clock admission time
    tokens  INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX reservations_tenant_ts ON reservations (tenant, ts);
CREATE INDEX reservations_ts ON reservations (ts);   -- for the global sweeper
-- PRAGMA journal_mode=WAL; busy_timeout=1000; synchronous=NORMAL
```

### Check-and-reserve is one transaction

```text
BEGIN IMMEDIATE                       -- take SQLite's write lock now, not at the first write
  now = clock()                       -- read *after* the lock: waiting must not leave "now" stale
  DELETE this tenant's rows with ts <= now - window
  used = SUM(tokens) for this tenant
  used + cost <= limit ? INSERT (id, tenant, now, cost) : compute Retry-After
COMMIT
```

`BEGIN IMMEDIATE` serialises writers across connections, threads and processes, so two requests can
never both read the same `used` and both insert. WAL lets readers run alongside the writer, and
`busy_timeout` makes a contending writer wait instead of failing. The event loop never touches
SQLite: every call runs on a worker thread through `anyio.to_thread.run_sync`. A limiter whose
database is unavailable fails closed with `503 rate_limiter_unavailable`.

### Availability when the database is stuck

- **Short busy timeout.** A transaction holds the write lock for well under a millisecond, so a
  writer waits at most `--db-busy-timeout-ms` (1000) for it. Waiting longer only delays the 503.
- **Fail fast.** Limiter calls queue for one worker thread per process. With a locked database each
  queued call used to wait out its own busy timeout, so the k-th caller got its 503 after k
  timeouts (5.6 s, 11.1 s, 16.6 s, ... with the old 5 s value). Now, after any SQLite error, every
  call fails immediately for `--db-fail-fast-ms` (1000), including the calls already queued, and
  the first call after that window probes the database again. A stuck database costs one caller
  one busy timeout; everyone else gets an immediate 503.
- **Cold start.** When several processes create a new database file at once, switching it to WAL
  needs an exclusive lock that `busy_timeout` does not cover, so an open could fail with "database
  is locked". Opening is retried with jittered backoff for up to 10 s on `SQLITE_BUSY` /
  `SQLITE_LOCKED`, and `llm-router` creates the database in the parent process before starting
  any worker. A database that cannot be opened at all stops start-up with a one-line error.

### Cost, reconciliation and release

- **Cost** = `prompt_estimate + max_tokens`. The estimate is `ceil(utf8_bytes / 4)` per message plus
  4 tokens of per-message overhead and 3 for reply priming. Counting UTF-8 bytes keeps CJK and emoji
  from being under-counted. `max_tokens` is always sent upstream (the default too), so the provider
  cannot generate more than was reserved.
- **Reconcile** on success: the row's `tokens` becomes the provider's `usage.total_tokens` (or
  prompt + completion). It records the truth even if that pushes the tenant over the limit (only
  possible when the prompt estimate was low). Later requests are then refused until the window
  drains.
- **Only plausible usage is trusted.** A correct provider reports between `1` and
  `max_tokens + 4 × prompt_estimate` tokens: the completion is capped by `max_tokens` (always sent
  upstream), and a token covers at least one byte, so the real prompt is at most 4× its
  `bytes / 4` estimate (the multiplied per-message overhead leaves room for chat-template tokens).
  A report outside that range (`0`, which some servers send when they do not count, negative,
  `10**15`, `2**63`, a provider that ignored `max_tokens`) or a missing one carries no information,
  so the reservation is simply kept: the tenant pays its own upper-bound estimate, never a free
  request and never a lock-out. The value is logged at WARNING and left out of the response's
  `usage`. Malformed usage fields are dropped rather than turning a valid completion into a bad
  response, so they never trigger a failover. *Why not clamp?* Clamping keeps part of a number
  already shown to be wrong: clamping `10**15` down to the ceiling would charge more than the
  reservation, and clamping `0` up to the prompt estimate would make the completion free.
- **The client always gets a completion that exists.** Any reconcile failure (database error,
  out-of-range value, anything else) is logged with its traceback and the row keeps its estimate.
- **Release** (DELETE) when no completion reaches the client: both providers failed, the request was
  rejected upstream, or the client went away. Policy: **a tenant is charged for the completion it
  receives**. A timed-out primary may have burned provider tokens, but the gateway cannot observe
  them, and billing tenants for a provider's slowness would tie quotas to upstream health. While a
  request is in flight its full reservation counts, so a tenant can never have more than its budget
  outstanding.
- Reservation ids are random UUIDs, not rowids. SQLite can reuse a deleted rowid, and a late
  reconcile must never land on someone else's row.
- Release and reconcile run in shielded cancel scopes, and the reservation itself is shielded, so a
  cancelled request cannot leak a reservation. If a reconcile fails anyway, the row keeps its
  upper-bound estimate: errors over-charge for at most one window, never under-charge.
- **Client disconnects.** uvicorn does not cancel a handler whose client hangs up; it only delivers
  `http.disconnect` on the ASGI receive channel. Routing therefore runs in a task group next to a
  watcher that awaits that message (the body has already been read, so it is the only one left).
  On disconnect the watcher cancels the group: the upstream attempt is abandoned (its connection
  closed), no failover starts, the reservation is released, an INFO line is logged and nothing is
  sent. When routing finishes first, the watcher is cancelled before the response goes out.

### Retry-After

Rows leave the window oldest first. The limiter walks the tenant's rows in `ts` order, adding up
tokens until they cover `used + cost - limit`. The row where that happens expires at
`ts + window`, so `Retry-After = ceil(ts + window - now)`. A request whose cost alone exceeds the
limit can never succeed, so it gets `413 request_too_large`, not a 429 with a meaningless wait.

### Eviction

- **Hot path:** each reservation deletes the requesting tenant's expired rows in the same
  transaction, so active tenants never accumulate rows and `SUM` only scans the live window.
- **Sweeper:** a background task deletes every expired row every 30 s, for keys that went quiet.
  Expired rows never count even before deletion (all reads filter on `ts > now - window`).

## Model router

Each attempt runs inside its own `anyio.move_on_after(deadline)` covering **everything**: pool wait,
connect, request, response headers, body and JSON parsing. httpx timeouts are per operation, so a
provider that drips one byte every 2 s would never trip them. The test suite includes that drip
case, over an in-process stream and over a real socket.

| Primary outcome | Gateway action | If it is the final outcome |
|-----------------|----------------|----------------------------|
| 2xx with a valid completion | return it (`X-Gateway-Route: primary`) | |
| 429 | fail over | `503 upstream_unavailable` (+ `Retry-After`) |
| deadline hit (no complete response in 3000 ms) | fail over | `504 upstream_timeout` |
| connect/read/protocol error | fail over (`--[no-]failover-on-connection-error`) | `503 upstream_unavailable` |
| 5xx | fail over (`--[no-]failover-on-server-error`) | `503 upstream_unavailable` (+ `Retry-After` from a 503) |
| 2xx with a malformed body, or a body over `--max-response-bytes` | fail over (with 5xx) | `502 upstream_bad_response` |
| 401 / 403 / 404 and any other 4xx not listed below | fail over (`--[no-]failover-on-refused`): bad provider key, wrong model or URL; another provider may serve it | `502 upstream_bad_response` (gateway misconfiguration, not the client's fault) |
| 408 / 409 / 425 | fail over (`--[no-]failover-on-refused`) | `503 upstream_unavailable` (transient) |
| 400 / 413 / 422 | **no** failover: the request itself is invalid, the secondary would refuse it too | `400 upstream_rejected_request` |

The secondary gets the same treatment with its own deadline. The error returned is decided by the
last attempt, and every attempt is logged.

- **Retry-After.** A provider's `Retry-After` on a 429 or 503 (delta-seconds or HTTP-date) is
  reduced to whole seconds in `[1, 60]`; anything else is dropped. When the gateway answers 503,
  it sends the smallest value among the providers that sent one: when the first of them expects
  to be available again. Only that integer is relayed, never the provider's text.
- **Response size cap.** Provider bodies are read in chunks (after content decoding) up to
  `--max-response-bytes` (4 MiB). A larger 2xx body is a bad response; the rest is not read, and
  the connection is dropped instead of being pooled.
- **Stale keep-alive connections.** A provider closing an idle keep-alive connection can race the
  gateway reusing it, and the request then fails with a read/protocol error although the provider
  is healthy. Pooled connections expire after 2 s, below the usual server keep-alive (5 s for
  uvicorn and Node, 75 s for nginx). And a request that went out on a *reused* connection and got
  no response at all (no status line) is sent once more, on a new connection, inside the same
  attempt deadline. httpcore's trace events tell the two cases apart: a request that opened its
  own connection, or whose response had started, is never resent, because the provider may have
  processed it. The residual risk is a provider that reads a request on a reused connection and
  then dies without answering: it may see the request twice. The tenant is still charged once
  (one reservation, one completion).

### Why a late primary response can never be used

- Attempts are **sequential**. When the deadline fires, the cancellation unwinds the
  `client.stream()` block, which closes the response; httpx drops a connection whose body was not
  fully read instead of returning it to the pool. Only then does the secondary start. Nothing from
  the primary exchange exists any more, so it cannot be returned, reconciled or double-counted.
- The result is used only if the whole block finished inside the deadline. A response that
  completes in the same instant the deadline fires is either fully used or fully discarded.
- Cancellation from *outside* (client gone, shutdown) is not a timeout. `move_on_after` only
  swallows its own cancellation, so outer cancellation propagates and never triggers a failover.
- Each upstream has its **own connection pool**, so a hanging primary that exhausts its pool cannot
  starve requests that are failing over.
- Only whitelisted fields are read from a provider response (`choices[].message.role/content`,
  `finish_reason`, `usage`). Provider extras such as `system_fingerprint` never reach clients.
- Upstream clients ignore proxy environment variables: egress is explicit configuration, and
  ambient `HTTPS_PROXY` must not be able to redirect provider traffic and API keys.

## Errors

```json
{"error": {"type": "rate_limit_error", "code": "tenant_token_limit_exceeded", "message": "...", "request_id": "req_..."}}
```

| HTTP | `type` | `code` | When |
|------|--------|--------|------|
| 400 | `invalid_request_error` | `invalid_json`, `invalid_request` | body is not JSON / fails validation (fields and rules named, values never echoed) |
| 400 | `invalid_request_error` | `streaming_not_supported` | `stream: true` |
| 400 | `invalid_request_error` | `upstream_rejected_request` | provider returned 400/413/422 |
| 401 | `authentication_error` | `invalid_api_key` | missing, malformed or unknown Bearer key (`WWW-Authenticate: Bearer`) |
| 404 / 405 | `not_found_error` / `invalid_request_error` | `not_found` / `method_not_allowed` | routing |
| 413 | `invalid_request_error` | `request_too_large` / `payload_too_large` | cost above the whole window / body too big |
| 429 | `rate_limit_error` | `tenant_token_limit_exceeded` | window full; `Retry-After` set |
| 502 | `upstream_error` | `upstream_bad_response` | malformed or oversized provider response, provider 401/403/404 |
| 503 | `upstream_error` | `upstream_unavailable` | providers rate limited, down, unreachable or 408/409/425; `Retry-After` when a provider sent one |
| 503 | `service_unavailable_error` | `rate_limiter_unavailable` | limiter database unavailable (fail closed, fast) |
| 504 | `timeout_error` | `upstream_timeout` | final attempt hit its deadline |
| 500 | `internal_error` | `internal_error` | anything unexpected (catch-all) |

- Messages are fixed strings written by the gateway; nothing from an upstream body, an exception or
  the request is interpolated into them.
- A pure ASGI middleware assigns a server-generated request id (client-supplied ids are ignored, as
  they could collide or inject text into logs), sets `X-Request-Id` on every response, and turns any
  unhandled exception into the generic 500. Starlette's 404/405 are mapped onto the same payload.
- Full details go only to the logs: upstream status, a bounded and escaped body snippet (escaping
  stops a provider from forging log lines), tracebacks. The request id is stored in a context
  variable, which follows the request into worker threads and appears on every log line.

## Concurrency and race conditions

| Hazard | Guard |
|--------|-------|
| Two requests both read `used` and both insert | `BEGIN IMMEDIATE`: check and insert under SQLite's write lock, across processes |
| Slow lock wait leaves a stale timestamp | clock read inside the transaction |
| Locked database: queued calls each wait a busy timeout | 1 s busy timeout, then every call fails fast for 1 s |
| Workers race to create a new database | open retried on `SQLITE_BUSY`; the parent creates it before workers start |
| Event loop blocked by disk I/O | SQLite only on worker threads (`anyio.to_thread`, capacity 1 per process) |
| Slow-drip body evades socket timeouts | one deadline over the whole attempt |
| Late primary response used or counted | sequential attempts; cancelled response closed before the secondary starts |
| Outer cancellation mistaken for a timeout | `move_on_after` only catches its own cancellation |
| Client hangs up, handler keeps routing and charges it | `http.disconnect` watcher cancels routing; reservation released, no failover |
| Cancelled request leaks its reservation | shielded reserve, release and reconcile |
| Late reconcile hits a reused row | UUID reservation ids |
| Primary pool exhaustion blocks failover | separate connection pool per upstream |
| Provider closes a keep-alive connection as it is reused | 2 s pool expiry; one resend only when nothing came back on a reused connection, inside the same deadline |
| A provider streams gigabytes within its deadline | body read in chunks, capped at `--max-response-bytes` |

## Mock provider

`mock-llm-provider --name NAME --port PORT --mode MODE [--delay SECONDS]`, switchable at runtime with
`PUT /_mock/mode {"mode": ..., "delay": ...}`; `GET /_mock/stats` counts requests and client
disconnects.

| Mode | Behaviour |
|------|-----------|
| `ok` | 200 completion; `usage` reports the full `max_tokens` as generated |
| `429` | 429 with `Retry-After: 1` and internal details in the body |
| `500` | 500 with a fake traceback, internal hostname and fake key |
| `400` | 400 `invalid_request_error` |
| `slow` | answers after `delay` seconds; stops early and counts a disconnect if the client hangs up |
| `drip` | headers at once, then the body in small pieces over `delay` seconds |

## Tests

```bash
pytest          # about 180 tests, ~20 s; deadlines are shortened to 250-300 ms to keep them fast
```

| File | Covers |
|------|--------|
| `test_limiter.py` | window sliding and boundaries, per-tenant and sweeper eviction, exact Retry-After, reconcile (incl. over limit, out-of-range counts refused), release, restart persistence, raw key absent from DB/WAL bytes, 8 threads and 4 processes on separate connections admitting exactly `floor(limit/cost)`, 12 connections creating a fresh database at once, a locked database failing fast (bounded timing) and recovering, a Hypothesis property checking every decision and Retry-After against the definition of the window |
| `test_main.py` | the database is created before any worker starts, an unusable path is a clean start-up error, the Windows multi-worker warning |
| `test_router.py` | 429/timeout/connect/5xx/malformed/oversized failover, refusals (401/403/404/408/...) fail over unless disabled, 400/413/422 not retried, policy switches, slow-drip body, real-socket checks that the timed-out primary sees the connection close, that a pooled connection closed by the provider is retried once and that a fresh connection or a started response is never resent, keep-alive expiry, upstream `Retry-After` relayed, late primary cancelled and never used, outer cancellation, 80 concurrent mixed requests |
| `test_errors.py` | catalogue, catch-all 500, routing errors, secrets/hostnames/tracebacks/provider and model names never in responses but present in logs with the same request id, Hypothesis property over random secrets |
| `test_api.py` | auth, headers, reconcile and release accounting (plausible usage recorded exactly; `2**63`, `10**15`, `0`, negative, malformed and missing usage keep the reservation and still serve; a failing reconcile still returns the completion), timeout race bounded by the deadline with only the fallback counted, client disconnect (receive channel only, no cancellation) and outer cancellation both release, exact Retry-After with an injected clock, validation without echoing input, 40 parallel requests admit exactly `floor(limit/cost)`, two app instances sharing one budget |
| `test_e2e.py` | uvicorn on ephemeral ports: slow and drip primaries fail over at the deadline and see the gateway hang up; a client that gives up after 0.5 s gets its primary attempt cancelled before the deadline, usage 0 and no secondary call; full failover / 503 / 429 walkthrough |

## Limitations and trade-offs

- **Clock skew.** Timestamps are wall-clock (`time.time()`) because they are shared between
  processes. SQLite must sit on a local disk (its locking is unreliable on network filesystems), so
  all workers share one host clock. The remaining risk is a wall-clock step (NTP), which can make
  rows expire early or late by the size of the step.
- **Estimate vs actual.** The pre-request estimate is a heuristic, not a tokenizer. A badly
  under-estimated prompt can let a tenant briefly exceed the limit; reconciliation records the real
  usage, so later requests pay for it.
- **Abandoned attempts are not billed** to the tenant (see the release policy). A provider may still
  charge the operator for them.
- **Single-host scale.** All writes for all tenants serialise on one SQLite write lock. That is
  plenty for one host (about 9,000 reserve + reconcile pairs per second on a developer laptop, one
  process, WAL mode). Multi-host deployments would move the same algorithm to Redis or Postgres.
- **No streaming**, and no upstream circuit breaker: a primary answering 429 (or 401) is tried
  again on the next request. Upstream `Retry-After` is relayed to clients on 503 but not used for a
  gateway-side cooldown.
- Upstream clients ignore `HTTP(S)_PROXY`; there is no option yet to configure an egress proxy.
