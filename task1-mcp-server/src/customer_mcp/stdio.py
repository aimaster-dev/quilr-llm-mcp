"""Hardened stdio transport.

Over stdio, stdout *is* the JSON-RPC wire: one message per line, nothing else. A single stray
`print()` corrupts the stream and the client drops the connection. Three layers keep it clean:

1. `claim.claim_stdio()`, the first thing `main()` does, serves the wire from private duplicates
   of fd 0 and fd 1 and points fd 1 at stderr (fd 0 at the null device) for the rest of the
   process's life. Output from C extensions, `os.write(1, ...)`, child processes and `atexit`
   hooks lands on stderr.
2. `sys.stdout` is rebound to `sys.stderr` by the same claim, so `print()` never touches the wire.
3. Logging is configured to stderr only (see `logging_config`).

Both claimed streams are handed to the SDK's `stdio_server()` explicitly, so it claims neither
descriptor itself and never points fd 1 back at the wire when serving ends.

Inbound framing is checked here, before the SDK sees a line. The SDK's stdin reader turns an
unparseable line into an exception that the dispatcher only logs. It also reads a request whose `id`
is invalid (`true`, `null`, `1.5`) as a *notification*, and one carrying both `method` and `error`
as a *response*. In each case the client never gets a reply and waits forever. JSON-RPC 2.0
requires an answer, so `check_frame` validates every raw line: malformed frames are answered with
`-32700 Parse error` / `-32600 Invalid Request`, and only messages the SDK's own parser accepts, as
the kind of message they claim to be, reach the SDK.

Frames are split on `\\n` only and each line is decoded as strict UTF-8 (a CRLF terminator is
tolerated because CR is JSON whitespace). The SDK's default text layer would instead split on a bare
CR and replace invalid UTF-8 with U+FFFD, silently altering a refund reason.

EOF on stdin does not end the session while requests are still being handled. The SDK cancels
in-flight handlers as soon as its input ends, so a client that writes its requests and closes stdin
(`customer-mcp < requests.jsonl`) would lose the replies to every request still running. The ids of
forwarded requests are tracked (`PendingRequests`) and the outbound stream clears each one as its
response or error goes out; at EOF the input is held open until nothing is pending (bounded by
`EOF_DRAIN_TIMEOUT`).
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import queue
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import IO, Any, cast

import anyio
import mcp_types as types
import pydantic_core
from anyio.abc import ObjectSendStream
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.dispatcher import as_request_id, coerce_request_id
from mcp.shared.message import SessionMessage
from pydantic import ValidationError

from customer_mcp.claim import StdioWire, claim_stdio

logger = logging.getLogger(__name__)

EOF_DRAIN_TIMEOUT = 30.0
"""Seconds to wait at stdin EOF for in-flight requests to be answered before shutting down."""


async def serve_stdio(server: Server[Any], wire: StdioWire | None = None) -> None:
    """Serve `server` over the claimed wire until stdin closes and every request is answered.

    `wire` is normally claimed by `main()` before anything else runs; it is claimed here otherwise.
    """
    wire = wire or claim_stdio()
    lines = CheckedLines(wire.reader)
    # newline="\n": no CRLF translation on Windows, every frame ends in exactly one LF.
    stdout = anyio.wrap_file(io.TextIOWrapper(wire.writer, encoding="utf-8", newline="\n"))
    try:
        # Both streams are explicit, so the SDK claims neither descriptor and, in particular, never
        # points fd 1 back at the wire when it exits. It only iterates `stdin` line by line, so a
        # checked line iterator can stand in for a file.
        async with stdio_server(stdin=cast("anyio.AsyncFile[str]", lines), stdout=stdout) as (
            read_stream,
            write_stream,
        ):
            lines.bind_replies(write_stream)
            # The server's replies pass through a stream that marks each answered request id. The
            # frame layer keeps the raw stream: its own replies answer nothing that was forwarded.
            outbound = AnsweringStream(write_stream, lines.pending)
            await server.run(read_stream, outbound, server.create_initialization_options())
    finally:
        lines.close()
        # The client sees EOF on stdout now rather than at process exit. Shielded: this also runs
        # when serving was cancelled, and the close must not be cut short.
        with anyio.CancelScope(shield=True), contextlib.suppress(OSError, ValueError):
            await stdout.aclose()


# ---------------------------------------------------------------------------------------------
# In-flight request tracking
# ---------------------------------------------------------------------------------------------
class PendingRequests:
    """Ids of requests handed to the SDK that have not been answered yet.

    A multiset: JSON-RPC does not forbid a client from reusing an id while the first request is
    still running, and each such request gets its own reply.
    """

    def __init__(self) -> None:
        self._ids: Counter[types.RequestId] = Counter()
        self._idle = anyio.Event()
        self._idle.set()

    def __len__(self) -> int:
        return self._ids.total()

    def add(self, request_id: types.RequestId) -> None:
        if self._idle.is_set():
            self._idle = anyio.Event()
        self._ids[request_id] += 1

    def answered(self, request_id: types.RequestId) -> None:
        self._discard(request_id)

    def cancelled(self, request_id: types.RequestId) -> None:
        """The client cancelled a request: the SDK will never answer it (MCP forbids a reply).

        The SDK matches `"7"` and `7` as the same id, so the cancellation is matched the same way.
        """
        if request_id in self._ids:
            self._discard(request_id)
            return
        target = coerce_request_id(request_id)
        for pending_id in self._ids:
            if coerce_request_id(pending_id) == target:
                self._discard(pending_id)
                return

    async def wait_until_idle(self) -> None:
        await self._idle.wait()

    def _discard(self, request_id: types.RequestId) -> None:
        if self._ids[request_id] <= 1:
            self._ids.pop(request_id, None)
        else:
            self._ids[request_id] -= 1
        if not self._ids:
            self._idle.set()


class AnsweringStream:
    """Outbound stream wrapper: each response or error the server sends answers one pending id.

    An id is cleared only once the reply has been handed to the stdout writer, so by the time the
    input is allowed to end, every reply is already on its way to the wire.
    """

    def __init__(self, inner: Any, pending: PendingRequests) -> None:
        self._inner = inner
        self._pending = pending

    async def send(self, item: SessionMessage, /) -> None:
        await self._inner.send(item)
        message = item.message
        if isinstance(message, types.JSONRPCResponse | types.JSONRPCError) and message.id is not None:
            self._pending.answered(message.id)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> AnsweringStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FrameCheck:
    """What to do with one inbound line: hand it to the SDK, drop it, or answer it with `error`."""

    forward: bool
    error: types.JSONRPCError | None = None
    # The decoded frame, kept for in-flight tracking; not part of the verdict.
    # The SDK's own parse of a forwarded frame, kept for in-flight tracking.
    message: types.JSONRPCMessage | None = field(default=None, compare=False)
    # Why a frame was dropped without a reply (a malformed response from the client).
    problem: str | None = None


DROP = FrameCheck(forward=False)

_JSON_WHITESPACE = " \t\r\n"
_MAX_ERROR_DETAIL = 200


def check_frame(line: bytes | str) -> FrameCheck:
    """Validate one newline-delimited frame before the SDK sees it.

    * blank line (JSON whitespace only): dropped (harmless framing noise)
    * not strict UTF-8, or not JSON as the SDK's parser reads it (lone surrogate escapes, nesting
      deeper than its recursion limit, the non-standard `NaN`/`Infinity` literals, trailing data):
      `-32700 Parse error`, `id: null`
    * a batch array: `-32600` (MCP removed JSON-RPC batching)
    * JSON that is not a valid request or notification: `-32600 Invalid Request`, echoing the `id`
      when it is a valid one. This covers a wrong `jsonrpc` version, a missing or non-string
      `method`, `method` combined with `result`/`error`, an `id` that is not a string or integer
      (MCP forbids `null`), non-object `params`, and anything else the SDK's message model rejects.
    * a response from the client (`result`/`error`, no `method`): forwarded when it is a valid
      JSON-RPC response, otherwise dropped with `problem` set. JSON-RPC never answers responses, and
      echoing its `id` would collide with the client's own request ids.

    Only frames that `types.jsonrpc_message_adapter` (the SDK's stdin parser, called the same way)
    accepts as the kind of message they claim to be are forwarded.
    """
    if isinstance(line, bytes):
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError:
            salvaged = _salvage_id(line.decode("utf-8", errors="replace"))
            return _reject(salvaged, types.PARSE_ERROR, "Parse error", "Invalid UTF-8")
    else:
        text = line
    if not text.strip(_JSON_WHITESPACE):
        return DROP
    try:
        # jiter, the parser behind the SDK's adapter, but without the NaN/Infinity extension.
        decoded = pydantic_core.from_json(text, allow_inf_nan=False)
    except (ValueError, TypeError) as exc:  # TypeError: a str holding an unpaired surrogate
        return _reject(_salvage_id(text), types.PARSE_ERROR, "Parse error", _json_problem(exc))

    if isinstance(decoded, list):
        return _reject(None, types.INVALID_REQUEST, "Invalid Request", "Batch requests are not supported")
    if not isinstance(decoded, dict):
        return _reject(None, types.INVALID_REQUEST, "Invalid Request", "Message must be a JSON object")
    if "method" not in decoded and ("result" in decoded or "error" in decoded):
        return _check_response(text, decoded)

    request_id = decoded.get("id")
    echo_id = request_id if _is_valid_id(request_id) else None
    problem = _envelope_problem(decoded)
    if problem is not None:
        return _reject(echo_id, types.INVALID_REQUEST, "Invalid Request", problem)
    try:
        message = _sdk_parse(text)
    except ValidationError as exc:
        if _is_json_error(exc):
            return _reject(echo_id, types.PARSE_ERROR, "Parse error", "Invalid JSON")
        return _reject(echo_id, types.INVALID_REQUEST, "Invalid Request", "Not a valid JSON-RPC request")
    expected = types.JSONRPCRequest if "id" in decoded else types.JSONRPCNotification
    if not isinstance(message, expected):  # defensive: the envelope checks should make this unreachable
        return _reject(echo_id, types.INVALID_REQUEST, "Invalid Request", "Not a valid JSON-RPC request")
    return FrameCheck(forward=True, message=message)


def _check_response(text: str, decoded: Mapping[str, Any]) -> FrameCheck:
    problem = _response_problem(decoded)
    if problem is None:
        try:
            message = _sdk_parse(text)
        except ValidationError:
            problem = "not a valid JSON-RPC response"
        else:
            if isinstance(message, types.JSONRPCResponse | types.JSONRPCError):
                return FrameCheck(forward=True, message=message)
            problem = "not a valid JSON-RPC response"
    return FrameCheck(forward=False, problem=problem)


def _sdk_parse(text: str) -> types.JSONRPCMessage:
    # Exactly the SDK's stdin reader call (mcp/server/stdio.py), so a forwarded line cannot fail there.
    return types.jsonrpc_message_adapter.validate_json(text, by_name=False)


def _envelope_problem(message: Mapping[str, Any]) -> str | None:
    if message.get("jsonrpc") != "2.0":
        return '"jsonrpc" must be exactly "2.0"'
    if "method" not in message:
        return '"method" is required'
    if not isinstance(message["method"], str):
        return '"method" must be a string'
    if "result" in message or "error" in message:
        return '"method" cannot be combined with "result" or "error"'
    if "id" in message and not _is_valid_id(message["id"]):
        return '"id" must be a string or an integer'
    if "params" in message and not isinstance(message["params"], dict):
        return '"params" must be an object'
    return None


def _response_problem(message: Mapping[str, Any]) -> str | None:
    if message.get("jsonrpc") != "2.0":
        return '"jsonrpc" must be exactly "2.0"'
    if "result" in message and "error" in message:
        return 'a response carries "result" or "error", not both'
    if "id" not in message:
        return '"id" is required'
    request_id = message["id"]
    if not (_is_valid_id(request_id) or (request_id is None and "error" in message)):
        return '"id" must be a string or an integer'
    return None


def _is_valid_id(value: object) -> bool:
    # bool is a subclass of int in Python, but `true` is not a valid JSON-RPC id.
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def _salvage_id(text: str) -> types.RequestId | None:
    """The id of a request the SDK cannot parse, when a lenient parser can still find it.

    A lone surrogate escape (which JavaScript's `JSON.stringify` emits for a split emoji), deep
    nesting or a stray invalid byte in a string make the whole frame unusable, but the client is
    still waiting on that request's id. JSON-RPC requires `id: null` only when the id cannot be
    detected, so the parse error carries it whenever it can be recovered unambiguously.
    """
    try:
        # strict=False also reads raw control characters inside strings.
        decoded = json.loads(text, strict=False)
    except (ValueError, RecursionError):
        return None
    if not isinstance(decoded, dict) or not isinstance(decoded.get("method"), str):
        return None  # only requests are answered
    request_id = decoded.get("id")
    if not _is_valid_id(request_id):
        return None
    if isinstance(request_id, str):
        if "�" in request_id:
            return None  # the id itself held invalid UTF-8: echoing a mangled id would mislead
        try:
            request_id.encode("utf-8")
        except UnicodeEncodeError:
            return None  # the id itself is an unpaired surrogate
    return cast("types.RequestId", request_id)


def _json_problem(exc: Exception) -> str:
    # jiter's messages name the defect and its position, never the input itself.
    if isinstance(exc, ValueError) and str(exc):
        return f"Invalid JSON: {exc}"[:_MAX_ERROR_DETAIL]
    return "Invalid JSON"


def _is_json_error(exc: ValidationError) -> bool:
    return any(error["type"] == "json_invalid" for error in exc.errors(include_url=False))


def _reject(
    request_id: types.RequestId | None, code: int, message: str, data: str | None = None
) -> FrameCheck:
    error = (
        types.ErrorData(code=code, message=message, data=data)
        if data
        else types.ErrorData(code=code, message=message)
    )
    return FrameCheck(forward=False, error=types.JSONRPCError(jsonrpc="2.0", id=request_id, error=error))


class CheckedLines:
    """Async line iterator handed to the SDK in place of stdin.

    Reads raw bytes and splits frames on `\\n` only. Well-formed frames are yielded, decoded, to
    the SDK's reader; malformed ones are answered directly on the outbound stream and never reach
    the dispatcher. At EOF the iterator ends only once every forwarded request has been answered,
    or after `drain_timeout` seconds.
    """

    def __init__(self, source: IO[bytes], *, drain_timeout: float | None = None) -> None:
        self._lines = LinePump(source)
        self._replies: ObjectSendStream[SessionMessage] | None = None
        self._bound = anyio.Event()
        self._drain_timeout = EOF_DRAIN_TIMEOUT if drain_timeout is None else drain_timeout
        self.pending = PendingRequests()

    def bind_replies(self, replies: ObjectSendStream[SessionMessage]) -> None:
        self._replies = replies
        self._bound.set()

    def close(self) -> None:
        self._lines.close()

    def __aiter__(self) -> CheckedLines:
        return self

    async def __anext__(self) -> str:
        while True:
            try:
                raw = await self._lines.readline()
            except OSError as exc:
                logger.error("reading stdin failed (%s); treating it as the end of input", exc)
                raw = b""
            if not raw:
                await self._drain()
                raise StopAsyncIteration
            check = check_frame(raw)
            if check.forward:
                self._track(check.message)
                return raw.decode("utf-8")  # cannot fail: check_frame decoded it strictly
            if check.error is not None:
                await self._answer(check.error)
            elif check.problem is not None:
                logger.warning("dropped malformed response from the client: %s", check.problem)

    def _track(self, message: types.JSONRPCMessage | None) -> None:
        if isinstance(message, types.JSONRPCRequest):
            self.pending.add(message.id)
        elif isinstance(message, types.JSONRPCNotification) and message.method == "notifications/cancelled":
            cancelled = as_request_id((message.params or {}).get("requestId"))
            if cancelled is not None:
                self.pending.cancelled(cancelled)

    async def _drain(self) -> None:
        """Hold the input open at EOF until every forwarded request has been answered."""
        if not self.pending:
            return
        logger.debug("stdin closed; waiting for %d in-flight request(s)", len(self.pending))
        with anyio.move_on_after(self._drain_timeout):
            await self.pending.wait_until_idle()
        if self.pending:
            logger.warning(
                "stdin closed; %d request(s) still unanswered after %.0f s, shutting down anyway",
                len(self.pending),
                self._drain_timeout,
            )

    async def _answer(self, reply: types.JSONRPCError) -> None:
        # The data string is a fixed description, never a copy of the client's input.
        logger.warning("rejected malformed frame: %s (%s)", reply.error.message, reply.error.data or "-")
        # The SDK starts reading before it hands out the outbound stream.
        await self._bound.wait()
        assert self._replies is not None
        with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
            await self._replies.send(SessionMessage(reply))


# ---------------------------------------------------------------------------------------------
# Reading stdin
# ---------------------------------------------------------------------------------------------
_READ_CHUNK = 64 * 1024
_MAX_QUEUED_LINES = 64
_WAKE = object()


class LinePump:
    """Reads `\\n`-terminated lines from a blocking byte stream in a daemon thread.

    A blocking read on a pipe cannot be cancelled. Done in one of anyio's worker threads, a read
    still pending when Ctrl+C arrives keeps the process alive until the client closes stdin: those
    are not daemon threads, and the interpreter waits for them at exit. This thread is a daemon, so
    it is simply left behind. It reads with `os.read`, holding no buffered-stream lock that
    interpreter shutdown could need. Lines are split on `\\n` only, so a bare CR stays in its frame.
    """

    def __init__(self, source: IO[bytes]) -> None:
        self._source = source
        # Bounded: when the server falls behind, the thread stops reading and the client blocks.
        self._queue: queue.Queue[object] = queue.Queue(_MAX_QUEUED_LINES)
        self._thread: threading.Thread | None = None
        self._finished = False

    async def readline(self) -> bytes:
        """The next line, including its `\\n`; the last bytes before EOF; `b""` at EOF."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="customer-mcp stdin", daemon=True)
            self._thread.start()
        item: object = _WAKE
        while item is _WAKE:  # a wake-up left over from an earlier cancelled read
            try:
                item = await anyio.to_thread.run_sync(self._queue.get, abandon_on_cancel=True)
            except anyio.get_cancelled_exc_class():
                # The worker thread is still blocked in get(): wake it, or it would hold up the exit.
                with contextlib.suppress(queue.Full):
                    self._queue.put_nowait(_WAKE)
                raise
        if isinstance(item, BaseException):
            self._finished = True
            raise item
        assert isinstance(item, bytes)
        if not item:
            self._finished = True
        return item

    def close(self) -> None:
        """Release the stream once the reader is done with it.

        While the thread may still be blocked in a read, the descriptor is left to the process exit:
        closed under the reader, it could be recycled for an unrelated file.
        """
        if self._thread is None or self._finished:
            with contextlib.suppress(OSError):
                self._source.close()

    def _run(self) -> None:
        try:
            try:
                fd = self._source.fileno()
            except (AttributeError, OSError, ValueError):  # an in-memory stream
                for line in iter(self._source.readline, b""):
                    self._queue.put(line)
            else:
                self._split_lines(fd)
            self._queue.put(b"")
        except Exception as exc:  # re-raised on the event loop side, where it can be logged
            self._queue.put(exc)

    def _split_lines(self, fd: int) -> None:
        pending = bytearray()
        while chunk := os.read(fd, _READ_CHUNK):
            start = 0
            search_from = len(pending)  # what is already pending holds no b"\n"
            pending += chunk
            while (end := pending.find(b"\n", search_from)) != -1:
                self._queue.put(bytes(pending[start : end + 1]))
                start = search_from = end + 1
            del pending[:start]
        if pending:
            self._queue.put(bytes(pending))
