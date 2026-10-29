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
| `--[no-]failover-on-server-error` | on | 5xx and malformed 2xx bodies |
| `--tokens-per-window`, `--window-seconds` | `50000`, `60` | per tenant API key |
| `--db-path` | `data/llm_router.db` | must be on a local disk (SQLite locking) |
| `--sweep-interval` | `30` | seconds between global evictions |
| `--default-max-tokens` | `1024` | used, and sent upstream, when the client omits `max_tokens` |
| `--max-body-bytes` | `1048576` | larger bodies get `413 payload_too_large` |
| `--host`, `--port`, `--workers`, `--log-level` | `127.0.0.1`, `8080`, `1`, `INFO` | |

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
| `Retry-After` | on 429: whole seconds until enough of the window drains for this request |
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
-- PRAGMA journal_mode=WAL; busy_timeout=5000; synchronous=NORMAL
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

### Cost, reconciliation and release

- **Cost** = `prompt_estimate + max_tokens`. The estimate is `ceil(utf8_bytes / 4)` per message plus
  4 tokens of per-message overhead and 3 for reply priming. Counting UTF-8 bytes keeps CJK and emoji
  from being under-counted. `max_tokens` is always sent upstream (the default too), so the provider
  cannot generate more than was reserved.
- **Reconcile** on success: the row's `tokens` becomes the provider's `usage.total_tokens` (or
  prompt + completion, or the estimate if usage is missing). It records the truth even if that
  pushes the tenant over the limit (only possible when the prompt estimate was low). Later requests
  are then refused until the window drains.
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
| 429 | fail over | `503 upstream_unavailable` |
| deadline hit (no complete response in 3000 ms) | fail over | `504 upstream_timeout` |
| connect/read/protocol error | fail over (configurable) | `503 upstream_unavailable` |
| 5xx | fail over (configurable) | `503 upstream_unavailable` |
| 2xx with a malformed body | fail over (configurable, with 5xx) | `502 upstream_bad_response` |
| 400 / 413 / 422 | **no** failover: the secondary would refuse it too | `400 upstream_rejected_request` |
| other 4xx (401, 403, 404, ...) | **no** failover | `502 upstream_bad_response` (gateway misconfiguration, not the client's fault) |

The secondary gets the same treatment with its own deadline. The error returned is decided by the
last attempt, and every attempt is logged.

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
| 502 | `upstream_error` | `upstream_bad_response` | malformed provider response, provider 401/403/404 |
| 503 | `upstream_error` | `upstream_unavailable` | providers rate limited, down or unreachable |
| 503 | `service_unavailable_error` | `rate_limiter_unavailable` | limiter database unavailable (fail closed) |
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
| Event loop blocked by disk I/O | SQLite only on worker threads (`anyio.to_thread`, capacity 1 per process) |
| Slow-drip body evades socket timeouts | one deadline over the whole attempt |
| Late primary response used or counted | sequential attempts; cancelled response closed before the secondary starts |
| Outer cancellation mistaken for a timeout | `move_on_after` only catches its own cancellation |
| Cancelled request leaks its reservation | shielded reserve, release and reconcile |
| Late reconcile hits a reused row | UUID reservation ids |
| Primary pool exhaustion blocks failover | separate connection pool per upstream |

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
| `test_limiter.py` | window sliding and boundaries, per-tenant and sweeper eviction, exact Retry-After, reconcile (incl. over limit), release, restart persistence, raw key absent from DB/WAL bytes, 8 threads and 4 processes on separate connections admitting exactly `floor(limit/cost)`, a Hypothesis property checking every decision and Retry-After against the definition of the window |
| `test_router.py` | 429/timeout/connect/5xx/malformed failover, policy switches, 4xx not retried, slow-drip body, real-socket check that the timed-out primary sees the connection close, late primary cancelled and never used, outer cancellation, 80 concurrent mixed requests |
| `test_errors.py` | catalogue, catch-all 500, routing errors, secrets/hostnames/tracebacks/provider and model names never in responses but present in logs with the same request id, Hypothesis property over random secrets |
| `test_api.py` | auth, headers, reconcile and release accounting, timeout race bounded by the deadline with only the fallback counted, exact Retry-After with an injected clock, validation without echoing input, 40 parallel requests admit exactly `floor(limit/cost)`, two app instances sharing one budget |
| `test_e2e.py` | uvicorn on ephemeral ports: slow and drip primaries fail over at the deadline and see the gateway hang up; full failover / 503 / 429 walkthrough |

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
- **No streaming**, and no upstream circuit breaker: a primary answering 429 is tried again on the
  next request. Upstream `Retry-After` is logged but not used for a cooldown.
- Upstream clients ignore `HTTP(S)_PROXY`; there is no option yet to configure an egress proxy.
