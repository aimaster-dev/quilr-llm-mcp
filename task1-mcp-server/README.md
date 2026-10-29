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

## Tools

| Tool | Arguments | Notes |
|------|-----------|-------|
| `get_customer_record` | `customer_id`: `CUST-` + exactly 5 digits | Read-only, idempotent |
| `trigger_refund` | `customer_id`; `amount`: number, `0 < amount <= 1,000,000`, finite, max 2 decimals; `reason`: 10-500 chars after trimming | Destructive, not idempotent |

Seed customers: `CUST-10001` to `CUST-10003` are active, `CUST-10004` is suspended and `CUST-10005` is
closed.

### Validation

The Pydantic v2 models in `schemas.py` are the single source of truth. They validate `tools/call`
arguments **and** generate the JSON Schema published by `tools/list`.

- **Strict mode, no coercion:** `"25.00"`, `true` and `null` are not numbers, and `10001` is not a
  customer ID.
- **`extra="forbid"`:** unknown keys (`customerId`, `admin: true`) are rejected, not silently ignored.
- **Customer ID** uses `[0-9]`, not `\d`, because `\d` matches Arabic-Indic and full-width digits.
  It is anchored so `"CUST-12345\n"` fails.
- **Amount** must be finite (no `NaN`/`Infinity`), with at most 2 decimal places (`0.1 + 0.2` is
  rejected). Money is then handled as `Decimal`.
- **Reason** is trimmed *before* the length check, so `"   damaged    "` cannot pass `min_length=10`.
  Control characters (ANSI escapes, NUL) are rejected; newlines and tabs are allowed.

### Two kinds of errors

| Failure | Reported as | Why |
|---------|-------------|-----|
| Unknown tool, malformed/missing/extra arguments | JSON-RPC error `-32602` *Invalid params*, field-level `data` | The request itself is invalid (protocol error) |
| Unknown customer, refund above balance, inactive account | Successful response with `isError: true` and `structuredContent.error` | The request was valid. The model should see the failure and adapt (MCP *tool execution error*) |

Validation errors list `field`, `type` and a generic `message` per problem. They never echo the
submitted values, which may contain PII or be arbitrarily large.

```json
{"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"Invalid arguments for tool 'trigger_refund'",
 "data":{"tool":"trigger_refund","errors":[
   {"field":"amount","type":"greater_than","message":"Input should be greater than 0"},
   {"field":"reason","type":"string_too_short","message":"String should have at least 10 characters"}]}}}
```

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
