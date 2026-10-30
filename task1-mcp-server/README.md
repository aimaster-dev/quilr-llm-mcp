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
| `trigger_refund` | `customer_id`; `amount`: number, `0 < amount <= 1,000,000`, finite, max 2 decimals; `reason`: 10-500 characters after trimming, no control or invisible format characters | Destructive, not idempotent |

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
- **Reason** is NFC-normalized, then trimmed, *then* length-checked, so `"   damaged    "` cannot
  pass the 10-character minimum. Lengths count Unicode code points of that normalized, trimmed
  value (an emoji is one, an `é` is one however it was encoded). Newlines and tabs are allowed.
  Rejected:
  - control characters (ANSI escapes, NUL, C1 controls);
  - invisible format characters (Unicode category `Cf`): zero-width spaces and BOMs, which would
    let a reason meet the minimum while showing nothing, bidi overrides and isolates, which make
    stored text display differently from what it contains, and tag characters, which can hide text
    from a human reviewer but not from a model. The exception is the zero-width joiner and
    non-joiner (U+200D, U+200C): Persian and Indic scripts and emoji sequences (👨‍👩‍👧) need them.
    They are allowed only between two visible characters, so they cannot pad a reason either.

### Published schema vs. enforced rules

`tools/list` states every rule JSON Schema can express, so a client that validates against the
`inputSchema` is not surprised by the server:

- `customer_id`: `pattern` `^CUST-[0-9]{5}$` plus `minLength`/`maxLength` 10 (redundant under
  ECMA-262, but Python's `re`, used by the `jsonschema` package, lets `$` match before a trailing
  newline);
- `amount`: `type: number`, `exclusiveMinimum: 0`, `maximum: 1000000`;
- `reason`: `minLength` 10, `maxLength` 500 and a `pattern` that rejects control and format
  characters anywhere and whitespace at either end. The server trims edge whitespace rather than
  rejecting it; the schema asks clients not to send it, so for any value the schema accepts,
  trimming changes nothing and the lengths mean exactly what the server checks. The pattern uses
  only `\uXXXX` escapes and character classes, which mean the same in ECMA-262 and Python's `re`.

Enforced by the server only, because JSON Schema cannot say them:

- **at most 2 decimal places** for `amount`. `multipleOf: 0.01` would be the obvious way, but
  validators test it in binary floating point, where `0.07` is not a multiple of `0.01`, so they
  would reject legitimate amounts;
- **NFC normalization** before the `reason` length check (`minLength` counts the raw code points,
  so the two differ only for text that is not already NFC);
- **format characters outside the Basic Multilingual Plane** (tag characters U+E0000-E007F and a
  few others): a JSON Schema pattern cannot name them portably, because ECMA-262 without the `u`
  flag sees UTF-16 surrogate pairs;
- **ZWJ/ZWNJ placement** (between two visible characters).

`tests/test_schemas.py` checks this list against the `jsonschema` package: outside these rules the
published schema and the server agree on every probe, and edge whitespace is the only input the
schema rejects but the server accepts.

### Two kinds of errors

| Failure | Reported as | Why |
|---------|-------------|-----|
| Unknown tool, malformed/missing/extra arguments | JSON-RPC error `-32602` *Invalid params*, field-level `data` | The request itself is invalid (protocol error) |
| Unknown customer, refund above balance, inactive account | Successful response with `isError: true`; the text content is `{"error":{"code":…,"message":…}}` | The request was valid. The model should see the failure and adapt (MCP *tool execution error*) |

Error results carry **no `structuredContent`**. Each tool's `outputSchema` describes its successful
result, and MCP says structured content MUST conform to it; a `{"error": …}` object would not, and
a client such as the TypeScript SDK validates any structured content it receives. Advertising a
`oneOf` of the record and an error shape would keep the check happy, but every consumer of a
successful result would then have to handle a schema that might be an error. So the error is a
machine-readable JSON document in the text content instead (codes: `customer_not_found`,
`customer_not_active`, `exceeds_refundable_balance`), which is also what the model reads:

```json
{"jsonrpc":"2.0","id":8,"result":{"isError":true,"content":[{"type":"text",
 "text":"{\"error\":{\"code\":\"customer_not_found\",\"message\":\"Customer CUST-99999 was not found\"}}"}]}}
```

Validation errors list `field`, `type` and a generic `message` per problem. They never echo the
submitted values, which may contain PII or be arbitrarily large. Names are echoed only when they
look like names: an unexpected argument key is reported by name if it is an identifier (`customerId`,
a likely typo) and as `(redacted)` otherwise (a key such as `SSN 123-45-6789`); an unknown tool name
only if it uses MCP's tool-name characters and is at most 128 long; an unknown method name likewise
(the SDK would echo it whole, 100 KB included). Malformed `tools/call` params that the SDK rejects
before the tool is looked up (`arguments` that is not an object, a non-string `name`) get the same
field-level `data`, where the SDK alone sends an empty string.

```json
{"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"Invalid arguments for tool 'trigger_refund'",
 "data":{"tool":"trigger_refund","errors":[
   {"field":"amount","type":"greater_than","message":"Input should be greater than 0"},
   {"field":"reason","type":"string_too_short","message":"String should have at least 10 characters"}]}}}
```

### JSON-RPC error mapping

| Situation | Code | `id` in the reply |
|-----------|------|-------------------|
| Line is not JSON as the SDK parses it: invalid UTF-8 (even inside a string), `NaN` / `Infinity`, lone surrogate escapes (`"\ud800"`), nesting deeper than ~200 levels, trailing data | `-32700` Parse error | the request's `id` if it can still be recovered, else `null` |
| Batch array (MCP removed JSON-RPC batching) | `-32600` Invalid Request | `null` |
| JSON but not a valid request: wrong `jsonrpc`, missing/non-string `method`, `method` combined with `result`/`error`, non-object `params` | `-32600` Invalid Request | echoed if valid |
| `id` that is not a string or integer (`true`, `null`, `1.5`, ...) | `-32600` Invalid Request | `null` |
| Unknown method | `-32601` Method not found | echoed |
| Unknown tool, malformed/missing/extra tool arguments | `-32602` Invalid params | echoed |
| Unexpected server fault | `-32603` Internal error. The traceback goes to stderr; the wire gets no internals | echoed |
| Blank line, notification, client response (a malformed one is logged on stderr) | no reply (JSON-RPC never answers these) | |

The first four rows are handled in `stdio.py` before the SDK sees the line. The SDK's stdin reader
only *logs* frames it cannot parse, reads a request with an invalid `id` as a notification, and reads
one carrying both `method` and `error` as a response. In each case the client would wait forever for
a reply. `check_frame` answers malformed frames itself and forwards a line only once the SDK's own
parser (`jsonrpc_message_adapter`, called exactly as the SDK calls it) accepts it as the kind of
message it claims to be.

Framing follows the MCP stdio transport: one message per line, split on `\n` only. A CRLF terminator
is accepted (CR is JSON whitespace), but a bare CR never splits a frame. Each line is decoded as
strict UTF-8, so invalid bytes are a parse error rather than a silent U+FFFD in a refund reason.

A parse error normally carries `id: null`, because the id could not be read. JSON-RPC requires that
only when the id cannot be detected, so when a lenient parse of the same line still finds a request
id (a lone surrogate from a JavaScript client that split an emoji, a stray invalid byte in a string),
the error echoes it and the client can match the error to the request it is waiting on.

## Use from an MCP host

```json
{
  "mcpServers": {
    "customer-service": { "command": "customer-mcp", "args": ["--log-level", "INFO"] }
  }
}
```

## stdout isolation

stdout is the wire, so the server claims it for the whole life of the process, as the very first
thing `main()` does, before logging is configured or the server is built (`claim.py`):

1. The wire is served from a private, non-inheritable duplicate of fd 1, and fd 1 is pointed at
   stderr (the Win32 standard handle too, which is what child processes inherit on Windows).
   Output from C extensions, `os.write(1, …)`, child processes and `atexit` hooks lands on stderr.
2. `sys.stdout` is rebound to `sys.stderr`, so a stray `print()` cannot reach the wire. Anything
   still buffered in the original `sys.stdout` is flushed *after* fd 1 was diverted, so it lands on
   stderr too, and so does anything written later to `sys.__stdout__` or through a library
   `StreamHandler` that captured the original stdout object.
3. `logging` is configured with a single stderr handler.

fd 1 is never pointed back at the wire, not even at exit. The SDK's own `stdio_server()` restores it
when serving ends, after which a buffered startup `print()` or a `sys.__stdout__` write was flushed
onto the wire behind the last reply. The claimed streams are passed to `stdio_server()` explicitly,
so it claims (and restores) nothing. Replies end in a single `\n` on every platform (no CRLF
translation on Windows).

stdin gets the same treatment: the server reads the wire from a private duplicate of fd 0 and points
fd 0 at the null device. A library or child process that reads stdin gets EOF and cannot consume
bytes meant for the server. With stdin closed altogether the server logs a warning and exits; with
stdout closed it refuses to start (exit code 1).

The entry point imports only the standard library before claiming the wire; the SDK, pydantic and
the server are imported afterwards, so their import-time output is diverted too. What no server can
take back is output written to the pipe before `main()` runs at all, for example by a wrapper
script that prints with `PYTHONUNBUFFERED` set, or a logging handler that flushes, before calling
`main()`. Output still sitting in Python's stdout buffer at that point is diverted.

The test suite checks all of this at the byte level, launching the server the way hosts do (stdout
fully buffered; `PYTHONUNBUFFERED` is removed unless a test asks for it). Every line the server
writes to stdout must parse as a JSON-RPC 2.0 message ending in `\n`, while `tests/noisy_server.py`
writes to stdout before `main()`, during startup, while serving (`print()`, fd 1, `sys.__stdout__`,
a library handler, a child process that also reads stdin) and from an `atexit` hook.

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
| `test_stdio_transport.py` | Handshake, stderr-only logging, stray-output isolation (startup, serving, exit; buffered and unbuffered), closed stdin/stdout, official SDK client over stdio |
| `test_shutdown.py` | Every request read before stdin EOF is answered (10 parallel replays), cancelled and stuck requests |
