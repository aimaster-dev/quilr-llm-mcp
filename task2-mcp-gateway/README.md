# Task 2 — MCP security gateway

A lightweight ASGI reverse proxy that sits between an AI agent and a downstream MCP server speaking
the **Streamable HTTP** transport.

## Run

```bash
pip install -e ".[dev]"
export MCP_GATEWAY_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
mcp-gateway --downstream-url http://127.0.0.1:9000/mcp --port 8000
mcp-gateway-token --subject alice --role viewer     # prints a bearer token
```

Every option can also be set as an environment variable (`MCP_GATEWAY_DOWNSTREAM_URL`, ...). The
token secret is read from `MCP_GATEWAY_JWT_SECRET` only, never from a flag, and the gateway refuses
to start if it is shorter than 32 bytes.

## Authentication

Every request needs `Authorization: Bearer <token>`, where the token is an **HS256 JWT**:

| Check | Rule |
|-------|------|
| Algorithm | Pinned to `HS256`. `none`, `HS512`, `RS256`/`ES256` (algorithm confusion) are rejected before any signature check |
| Claims | `exp`, `iat`, `iss`, `aud`, `sub`, `role` are required. `exp`/`nbf`/`iat` are checked with 30 s leeway |
| `iss` / `aud` | Must equal `--jwt-issuer` / `--jwt-audience`, so tokens minted for other services are refused |
| `role` | Exactly `admin` or `viewer`. `Admin`, `root`, `["admin"]` are rejected |
| `sub` | 1-256 printable ASCII characters (it is forwarded in a header, so CR/LF injection is impossible) |

Failures follow RFC 6750, with a JSON-RPC error body (`-32003`):

| Situation | Status | `WWW-Authenticate` |
|-----------|--------|--------------------|
| No `Authorization` header, or another scheme | `401` | `Bearer realm="mcp-gateway"` |
| Bad signature, expired, wrong `iss`/`aud`/`role`, ... | `401` | `... error="invalid_token", error_description="..."` |
| Malformed header (`Bearer a b`, two headers) | `400` | `... error="invalid_request", ...` |

The client only learns "invalid" or "expired". The precise reason (`InvalidSignatureError`,
`InvalidAudienceError`, ...) goes to the log, and the token itself is never logged.

**No token passthrough.** The client's `Authorization` (and `Cookie`) header is not forwarded: the
MCP spec forbids passing a token through to a service it was not issued for. The gateway sends the
verified identity instead, as `X-Gateway-Subject` and `X-Gateway-Role`. Any `X-Gateway-*` header
sent by the client is dropped first, so it cannot be spoofed.

## Proxying

| Client request | Gateway behaviour |
|----------------|-------------------|
| `POST /mcp` | Requires `Content-Type: application/json` (`415` otherwise) and a body within `--max-body-bytes` (`413`). The response, JSON or SSE, is streamed back chunk by chunk. |
| `GET /mcp` | Server-to-client SSE stream, relayed with no read timeout. |
| `DELETE /mcp` | Session termination, relayed. |
| anything else | `405` with an `Allow` header. |

- Hop-by-hop headers (`Connection` and the headers it names, `Keep-Alive`, `TE`, `Upgrade`, ...) are
  dropped in both directions. `Mcp-Session-Id`, `MCP-Protocol-Version` and `Last-Event-ID` pass
  through unchanged.
- The downstream is asked for `Accept-Encoding: identity`, so an SSE stream is never compressed
  (a compressor buffers events).
- If the downstream cannot be reached the client gets `502`, and `504` if it times out, each with a
  JSON-RPC error body instead of a stack trace.
