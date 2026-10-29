# Task 1 — Customer MCP server (stdio)

An MCP server built on the official Python SDK (`mcp` v2) that serves customer-service tools over
the **stdio** transport.

## Run

```bash
pip install -e ".[dev]"
customer-mcp --log-level DEBUG        # or: python -m customer_mcp
```

The process reads JSON-RPC from stdin and writes JSON-RPC to stdout, one message per line.
**Logs go to stderr only.**

## Use from an MCP host

```json
{
  "mcpServers": {
    "customer-service": { "command": "customer-mcp", "args": ["--log-level", "INFO"] }
  }
}
```

## stdout isolation

stdout is the wire, so the server protects it in three layers:

1. The SDK's `stdio_server()` serves the wire from a private duplicate of fd 1 and points fd 1 at
   stderr. Output from C extensions, `os.write(1, …)` and child processes cannot reach the client.
2. `sys.stdout` is rebound to `sys.stderr` while serving, so a stray `print()` cannot reach the
   wire either.
3. `logging` is configured with a single stderr handler.

The test suite checks this at the byte level: every line the server writes to stdout must parse
as a JSON-RPC 2.0 message.
