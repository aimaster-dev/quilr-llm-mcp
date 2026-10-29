# Task 2 — MCP security gateway

A lightweight ASGI reverse proxy that sits between an AI agent and a downstream MCP server speaking
the **Streamable HTTP** transport.

## Run

```bash
pip install -e ".[dev]"
mcp-gateway --downstream-url http://127.0.0.1:9000/mcp --port 8000
```

Every option can also be set as an environment variable (`MCP_GATEWAY_DOWNSTREAM_URL`, ...).

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
