# Task 2 — MCP security gateway

A lightweight reverse proxy (Starlette + httpx, served by uvicorn) that sits between an AI agent and
a downstream MCP server speaking the **Streamable HTTP** transport. It authenticates every request
with a Bearer JWT, reads the caller's role from it, and refuses `admin_*` tool calls from anyone
who is not an `admin`, **without ever contacting the downstream server**.

```
  AI agent (MCP client)              mcp-gateway                                 downstream MCP server
                                     +-----------------------------------+
  POST /mcp                          | 1. authenticate: JWT -> role      |
  Authorization: Bearer <jwt>  ----> | 2. Content-Type, body size        |
                                     | 3. strict JSON-RPC parse          |  POST /mcp (canonical body)
                                     | 4. authorize (policy.py)          |  X-Gateway-Subject / -Role
                                     | 5. one audit line                 |  no Authorization header
  JSON or SSE, streamed  <---------- | 6. forward, stream the response   | ----------------------------->
                                     +-----------------------------------+ <----------------------------
          ^                                         |
          +---- 401 / 4xx / -32001 answered by the gateway itself (downstream never contacted)
```

## Quick start: a curl walkthrough

```bash
pip install -e ".[dev]"
export MCP_GATEWAY_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"

mock-mcp-server --port 9000 &                                           # downstream, see below
mcp-gateway --downstream-url http://127.0.0.1:9000/mcp --port 8000 &    # the gateway

VIEWER=$(mcp-gateway-token --subject alice --role viewer)
ADMIN=$(mcp-gateway-token --subject root --role admin)
mcp() {  # usage: mcp <token> <json-rpc message>
  curl -s http://127.0.0.1:8000/mcp -H "Authorization: Bearer $1" \
    -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -d "$2"; echo
}
```

(On Windows use Git Bash, or PowerShell with `$env:MCP_GATEWAY_JWT_SECRET = ...` and `curl.exe`.)

**`tools/list` is forwarded transparently**, for every role:

```bash
mcp "$VIEWER" '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
# {"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"get_weather",...},{"name":"admin_reset_key",...},...]}}
```

**Ordinary tools work for a viewer:**

```bash
mcp "$VIEWER" '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_weather","arguments":{"city":"Paris"}}}'
# {"jsonrpc":"2.0","id":2,"result":{"content":[...],"isError":false,"structuredContent":{"city":"Paris","forecast":"sunny","temperature_c":21}}}
```

**An admin tool is refused by the gateway** (the mock server logs nothing: it never sees the call):

```bash
mcp "$VIEWER" '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"key_id":"k-1"}}}'
# {"jsonrpc":"2.0","id":3,"error":{"code":-32001,"message":"Unauthorized Tool Call","data":{"requiredRole":"admin"}}}
```

**...and allowed for an admin:**

```bash
mcp "$ADMIN" '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"key_id":"k-1"}}}'
# {"jsonrpc":"2.0","id":4,"result":{...,"structuredContent":{"key_id":"k-1","status":"rotated"}}}
```

**The downstream sees the verified identity, never the token:**

```bash
mcp "$VIEWER" '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"whoami","arguments":{}}}'
# {..."structuredContent":{"subject":"alice","role":"viewer","authorization_forwarded":false}}}
```

**Bypass attempts fail closed:**

```bash
mcp "$VIEWER" '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"ADMIN_reset_key","arguments":{"key_id":"k-1"}}}'
# {"jsonrpc":"2.0","id":6,"error":{"code":-32001,"message":"Unauthorized Tool Call","data":{"requiredRole":"admin"}}}
mcp "$VIEWER" '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"get_weather","name":"admin_reset_key"}}'
# {"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error: duplicate object key"}}
mcp "$VIEWER" '[{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"admin_reset_key"}}]'
# {"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"Invalid Request: batch requests are not supported"}}
```

**No token, no entry:**

```bash
curl -si http://127.0.0.1:8000/mcp -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":9,"method":"tools/list"}'
# HTTP/1.1 401 Unauthorized
# www-authenticate: Bearer realm="mcp-gateway"
# {"jsonrpc":"2.0","id":null,"error":{"code":-32003,"message":"A bearer token is required"}}
```

The official Python SDK works the same way; pass the token on its HTTP client:

```python
async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as http:
    async with Client(streamable_http_client("http://127.0.0.1:8000/mcp", http_client=http)) as client:
        await client.call_tool("admin_reset_key", {"key_id": "k-1"})  # viewer: MCPError(code=-32001)
```

## Authentication

Every request needs `Authorization: Bearer <token>`, where the token is an **HS256 JWT**:

| Check | Rule |
|-------|------|
| Algorithm | Pinned to `HS256`. `none`, `HS512`, `RS256`/`ES256` (algorithm confusion) are rejected before any signature check |
| Claims | `exp`, `iat`, `iss`, `aud`, `sub`, `role` are required. `exp`/`nbf`/`iat` are checked with 30 s leeway |
| `iss` / `aud` | Must equal `--jwt-issuer` / `--jwt-audience`, so tokens minted for other services are refused |
| `role` | Exactly `admin` or `viewer`. `Admin`, `root`, `["admin"]` are rejected |
| `sub` | 1-256 printable ASCII characters. It is forwarded in a header, so CR/LF injection is impossible |

Failures follow RFC 6750, with a JSON-RPC error body (`-32003`):

| Situation | Status | `WWW-Authenticate` |
|-----------|--------|--------------------|
| No `Authorization` header, or another scheme | `401` | `Bearer realm="mcp-gateway"` |
| Bad signature, expired, wrong `iss`/`aud`/`role`, ... | `401` | `... error="invalid_token", error_description="..."` |
| Malformed header (`Bearer a b`, two headers) | `400` | `... error="invalid_request", ...` |

The client only learns "invalid" or "expired". The precise reason (`InvalidSignatureError`,
`InvalidAudienceError`, ...) goes to the audit log, and the token itself is never logged.

**No token passthrough.** The client's `Authorization` (and `Cookie`) header is not forwarded: the
MCP spec forbids passing a token through to a service it was not issued for. The gateway sends the
verified identity instead, as `X-Gateway-Subject` and `X-Gateway-Role`. Any `X-Gateway-*` header
sent by the client is dropped first, so it cannot be spoofed.

## Authorization

Only `tools/call` is gated. Every other method (`initialize`, `tools/list`, `ping`,
`notifications/*`, `resources/*`, ...) and every client response is forwarded unchanged.

| `tools/call` with `params.name` | `admin` | `viewer` |
|---------------------------------|---------|----------|
| `get_weather` | forwarded | forwarded |
| `admin_reset_key`, `admin_delete_user`, ... | forwarded | `-32001` *Unauthorized Tool Call*, **downstream never contacted** |

The denial reuses the request's `id` and is sent with HTTP `200`. Why not `403`: the HTTP exchange
worked, and only this one call was refused. MCP clients surface it as an ordinary JSON-RPC error
for that call (the official SDK raises `MCPError(code=-32001)`) and the session carries on. A
`401`/`403` would look like a broken session, or make an OAuth-aware client try to refresh its
token.

### Failing closed

The gateway decides on its own parse of the request, so it refuses anything another parser (the
downstream's) could read differently:

| Attempt | Result |
|---------|--------|
| `Admin_reset_key`, `" admin_reset_key"`, a zero-width space or BOM before `admin_`, full-width `ａｄｍｉｎ_x` | `-32001`. The prefix check runs on a normalised name: NFKC, case-folded, whitespace and invisible characters removed |
| `"method": "Tools/Call"` | Gated like `tools/call` (same normalisation) |
| Duplicate keys: `{"name":"get_weather","name":"admin_x"}` | `-32700`: parsers disagree on which key wins |
| Batch `[{...tools/list}, {...admin_reset_key}]` | `-32600`: MCP 2025-06-18 dropped batching, so nothing can hide in one |
| `tools/call` notification (no `id`) | `-32600`: a notification cannot receive a denial, so it is never forwarded unchecked |
| `params.name` missing, `null`, a number, an array; `{"name":..., "Name":...}` | `-32602` |
| `NaN`/`Infinity`, `1e999`, a lone surrogate (U+D800), invalid UTF-8, UTF-16, a BOM | `-32700` |
| An `Mcp-Name` / `Mcp-Method` header that contradicts the body (MCP 2026-07-28 routing headers) | `-32020` |

The gateway then forwards **the message it authorized**, re-serialised from the parsed object as
compact UTF-8 JSON, not the bytes it received. Whatever escapes or whitespace the client used, the
downstream parses exactly what the gateway checked. The original tool name is not rewritten: an
admin calling `Admin_reset_key` reaches the downstream, which decides whether that tool exists.

The rules live in `policy.py` and `jsonrpc.py`: pure functions, no HTTP, unit- and property-tested.

### Audit log

Each request produces exactly one JSON line on the `mcp_gateway.audit` logger (stderr): `allow`
(forwarded), `deny` (-32001) or `reject` (unauthenticated or malformed).

```json
{"ts":"2026-10-29T09:12:44.913+00:00","event":"mcp_gateway.decision","decision":"deny","reason":"admin_tool_requires_admin_role","request_id":"5f0c...","subject":"alice","role":"viewer","http_method":"POST","method":"tools/call","rpc_id":3,"tool":"admin_reset_key","status":200,"code":-32001}
```

The token is never logged. Client-supplied strings are JSON-escaped and truncated, so a tool name
containing a newline cannot forge a second line. The `request_id` is also sent downstream as
`X-Gateway-Request-Id` for correlation. Audit lines are emitted even when `--log-level` is
`WARNING`.

## Proxying

| Client request | Gateway behaviour |
|----------------|-------------------|
| `POST /mcp` | Requires `Content-Type: application/json` (`415` otherwise) and a body within `--max-body-bytes` (`413`, enforced while reading, so a huge body is never buffered). The response, JSON or SSE, is streamed back chunk by chunk. |
| `GET /mcp` | Server-to-client SSE stream, relayed with no read timeout. |
| `DELETE /mcp` | Session termination, relayed. |
| anything else | `405` with an `Allow` header. |

- Hop-by-hop headers (`Connection` and the headers it names, `Keep-Alive`, `TE`, `Upgrade`, ...) are
  dropped in both directions. `Mcp-Session-Id`, `MCP-Protocol-Version` and `Last-Event-ID` pass
  through unchanged.
- The downstream is asked for `Accept-Encoding: identity`, so an SSE stream is never compressed
  (a compressor buffers events).
- The relay watches for client disconnects and always closes the downstream response, so an
  abandoned SSE stream releases its downstream connection immediately.
- The downstream URL is fixed configuration: the client's path and query string are not forwarded,
  and proxy environment variables (`HTTP_PROXY`, ...) are ignored.

## Error reference

| HTTP | JSON-RPC `error.code` | Cause |
|------|-----------------------|-------|
| `200` | `-32001` Unauthorized Tool Call | `admin_*` tool called without the `admin` role |
| `401` / `400` | `-32003` | Missing, invalid or expired token / malformed `Authorization` header |
| `400` | `-32700` Parse error | Not JSON, duplicate keys, `NaN`, lone surrogates, not UTF-8 |
| `400` | `-32600` Invalid Request | Bad envelope (`jsonrpc`, `id`, `method`, extra members), batch, `tools/call` notification |
| `400` | `-32602` Invalid params | `params.name` missing, not a string or ambiguous; positional `params` |
| `400` | `-32020` | `Mcp-Method`/`Mcp-Name` header contradicts the body |
| `405` / `413` / `415` | `-32600` | Wrong HTTP method / body too large / not `application/json` |
| `502` | `-32004` | Downstream unreachable or broken response |
| `504` | `-32005` | Downstream timed out |

Errors the gateway produces echo the request `id` once it is known, and never echo input values.
Errors produced by the downstream are relayed as they are.

## Configuration

| Flag | Environment variable | Default |
|------|----------------------|---------|
| (none: never a flag) | `MCP_GATEWAY_JWT_SECRET` | **required**, at least 32 bytes |
| `--downstream-url` | `MCP_GATEWAY_DOWNSTREAM_URL` | `http://127.0.0.1:9000/mcp` |
| `--host` / `--port` | `MCP_GATEWAY_HOST` / `MCP_GATEWAY_PORT` | `127.0.0.1` / `8000` |
| `--jwt-issuer` / `--jwt-audience` | `MCP_GATEWAY_JWT_ISSUER` / `MCP_GATEWAY_JWT_AUDIENCE` | `mcp-gateway-dev` / `mcp-gateway` |
| `--jwt-leeway` | `MCP_GATEWAY_JWT_LEEWAY` | `30` seconds |
| `--connect-timeout` | `MCP_GATEWAY_CONNECT_TIMEOUT` | `5` seconds (`504` when exceeded) |
| `--read-timeout` | `MCP_GATEWAY_READ_TIMEOUT` | `60` seconds of downstream silence on POST/DELETE |
| `--max-body-bytes` | `MCP_GATEWAY_MAX_BODY_BYTES` | `1048576` |
| `--log-level` | `MCP_GATEWAY_LOG_LEVEL` | `INFO` |

Settings are validated at startup: a short secret, a non-http(s) downstream URL or a URL with
embedded credentials stop the gateway before it binds a port. `mcp-gateway-token` reads the same
secret, issuer and audience variables.

## Mock downstream server

`mock-mcp-server` is a real MCP server built on the official Python SDK (`mcp` v2), with the tools
`get_weather`, `admin_reset_key`, `admin_delete_user` and `whoami`. By default it is stateless and
answers with JSON (easy to curl); `--sse` streams every response as Server-Sent Events and
`--stateful` issues `Mcp-Session-Id`s. It records every call it receives, which is how the tests
prove that a denied call never arrived.

## Tests

```bash
pytest   # ~300 tests, a few seconds
```

| File | Covers |
|------|--------|
| `test_policy.py`, `test_jsonrpc.py` | The pure core: name normalisation (with Hypothesis properties), `authorize`, routing headers, strict parsing, canonical form |
| `test_auth.py` | `alg: none`, HS512 and RS256/ES256 confusion, tampered payloads, `exp`/`nbf`/`iat`, `iss`/`aud`, roles, subjects, RFC 6750 challenges, no token passthrough, spoofed identity headers, the token CLI |
| `test_authorization.py` | `-32001` over HTTP against an in-memory downstream that records requests (it is never contacted on denial), bypass attempts, canonical forwarding, audit lines |
| `test_proxy.py` | Forwarding, hop-by-hop headers, session headers, unbuffered SSE, disconnect handling, `405`/`413`/`415`/`502`/`504` |
| `test_e2e.py` | Official SDK client -> gateway -> mock server over real HTTP (uvicorn on ephemeral ports), both roles, JSON and SSE, the 2026-07-28 protocol and the session-based handshake |

## Security notes and limits

- **The downstream must only be reachable through the gateway** (bind it to localhost or a private
  network). It trusts `X-Gateway-*` headers because nothing else can reach it. Signing the asserted
  identity (a short-lived gateway token, or mTLS) would remove that assumption.
- **HS256 means a shared secret**: whoever holds it can mint tokens. In production an identity
  provider would issue RS256/ES256 tokens and the gateway would verify them against its JWKS,
  holding public keys only. The pinned-algorithm verification stays the same.
- **`tools/list` is not filtered**, as the task requires: viewers can see the admin tools, they just
  cannot call them. Filtering per role would mean rewriting responses, including SSE streams.
- **Sessions are not bound to identities**, but every request is authorized on its own token, so a
  viewer who learns an admin's `Mcp-Session-Id` still cannot call an admin tool.
- Server-to-client traffic (responses, SSE events, `GET` streams) is relayed without inspection.
- Not covered: rate limiting, `Origin` allowlisting (a page in a browser cannot present the bearer
  token, so DNS rebinding alone gets no access), and OAuth protected-resource metadata.
