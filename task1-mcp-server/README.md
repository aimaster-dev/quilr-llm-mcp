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

Closing stdin ends the session, but not before every request already read has been answered: a
client may write its whole session and close stdin (`customer-mcp < requests.jsonl`) and still get
every reply. The server waits up to 30 seconds for in-flight requests, logs a warning on stderr if
any are still running, then shuts down (those get a `Connection closed` error). Requests the client
cancelled with `notifications/cancelled` are not waited for, since MCP forbids answering them.

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

### JSON-RPC error mapping

| Situation | Code | `id` in the reply |
|-----------|------|-------------------|
| Line is not JSON (incl. invalid UTF-8 and the non-standard `NaN` / `Infinity` literals) | `-32700` Parse error | `null` |
| Batch array (MCP removed JSON-RPC batching) | `-32600` Invalid Request | `null` |
| JSON but not a valid request: wrong `jsonrpc`, missing/non-string `method`, non-object `params` | `-32600` Invalid Request | echoed if valid |
| `id` that is not a string or integer (`true`, `null`, `1.5`, ...) | `-32600` Invalid Request | `null` |
| Unknown method | `-32601` Method not found | echoed |
| Unknown tool, malformed/missing/extra tool arguments | `-32602` Invalid params | echoed |
| Unexpected server fault | `-32603` Internal error. The traceback goes to stderr; the wire gets no internals | echoed |
| Blank line, client response, notification | no reply (JSON-RPC never answers these) | |

The first four rows are handled in `stdio.py` before the SDK sees the line. The SDK's stdin reader
only *logs* frames it cannot parse, and it reads a request with an invalid `id` as a notification. In
both cases the client would wait forever for a reply. `check_frame` validates the JSON-RPC envelope
of every raw line and answers malformed frames itself, so only well-formed messages reach the SDK.

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

stdin gets the same treatment: the server reads the wire from a private duplicate of fd 0 and points
fd 0 at the null device. A library or child process that reads stdin gets EOF and cannot consume
bytes meant for the server.

The test suite checks all of this at the byte level. Every line the server writes to stdout must
parse as a JSON-RPC 2.0 message, including when a tool handler `print()`s, writes to fd 1 directly
and spawns a child process that writes to stdout and reads stdin (`tests/noisy_server.py`).

## Tests

```bash
pytest
```

| File | Covers |
|------|--------|
| `test_schemas.py` | Validation edge cases for every field |
| `test_tools.py` | Tool behaviour through the SDK client (in-process), incl. concurrent refunds |
| `test_frames.py` | Envelope validation of raw stdin lines |
| `test_protocol_errors.py` | Every JSON-RPC error code, asserted on the real binary's stdout |
| `test_stdio_transport.py` | Handshake, stderr-only logging, stray-output isolation, official SDK client over stdio |
| `test_shutdown.py` | Every request read before stdin EOF is answered (10 parallel replays), cancelled and stuck requests |
