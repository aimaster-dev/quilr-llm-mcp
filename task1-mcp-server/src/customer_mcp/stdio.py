"""Hardened stdio transport.

Over stdio, stdout *is* the JSON-RPC wire: one message per line, nothing else. A single stray
`print()` corrupts the stream and the client drops the connection. Three layers keep it clean:

1. The SDK's `stdio_server()` claims fd 1. It serves the wire from a private duplicate and
   points fd 1 at stderr, so output from C extensions, `os.write(1, ...)` and child processes
   lands on stderr.
2. `sys.stdout` is rebound to `sys.stderr` while serving. `print()` never touches the wire,
   even on platforms where the fd claim falls back to sharing the original stdout buffer.
3. Logging is configured to stderr only (see `logging_config`).

Inbound framing is checked here, before the SDK sees a line. The SDK's stdin reader turns an
unparseable line into an exception that the dispatcher only logs. It also reads a request whose `id`
is invalid (`true`, `null`, `1.5`) as a *notification*. Either way the client never gets a reply
and waits forever. JSON-RPC 2.0 requires an answer, so `check_frame` validates the envelope of every
raw line: malformed frames are answered with `-32700 Parse error` / `-32600 Invalid Request`, and
only well-formed messages reach the SDK.

Reading stdin ourselves means we also own its claim (`_claim_stdin`). As the SDK does for fd 1,
the wire is read from a private duplicate and fd 0 points at the null device, so nothing else in
the process (or a child process) can consume bytes meant for the server.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import IO, Any, cast

import anyio
import mcp_types as types
from anyio.abc import ObjectSendStream
from mcp.os.win32.utilities import rebind_std_handle_to_fd
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage

logger = logging.getLogger(__name__)


async def serve_stdio(server: Server[Any]) -> None:
    """Serve `server` over stdin/stdout until the client closes stdin."""
    with _claim_stdin() as wire_in:
        lines = CheckedLines(
            anyio.wrap_file(_UnownedTextWrapper(wire_in, encoding="utf-8", errors="replace"))
        )
        # Order matters: the SDK must claim the *real* stdout before sys.stdout is rebound.
        # The SDK only iterates `stdin` line by line, so a checked line iterator can stand in for it.
        async with stdio_server(stdin=cast("anyio.AsyncFile[str]", lines)) as (read_stream, write_stream):
            lines.bind_replies(write_stream)
            with contextlib.redirect_stdout(sys.stderr):
                await server.run(read_stream, write_stream, server.create_initialization_options())


# ---------------------------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FrameCheck:
    """What to do with one inbound line: hand it to the SDK, drop it, or answer it with `error`."""

    forward: bool
    error: types.JSONRPCError | None = None


FORWARD = FrameCheck(forward=True)
DROP = FrameCheck(forward=False)


def check_frame(line: str) -> FrameCheck:
    """Validate one newline-delimited frame against the JSON-RPC 2.0 envelope MCP uses.

    * blank line: dropped (harmless framing noise)
    * not JSON (including the non-standard `NaN`/`Infinity` literals): `-32700 Parse error`, `id: null`
    * a batch array: `-32600` (MCP removed JSON-RPC batching)
    * JSON that is not a valid request or notification: `-32600 Invalid Request`, echoing the `id`
      when it is a valid one. This covers a wrong `jsonrpc` version, a missing or non-string `method`,
      an `id` that is not a string or integer (MCP forbids `null`), and non-object `params`.
    * a response from the client (`result`/`error`, no `method`): forwarded. JSON-RPC never answers
      responses, so the SDK decides what to do with it.
    """
    if not line.strip():
        return DROP
    try:
        message = json.loads(line, parse_constant=_reject_non_standard_constant)
    except (ValueError, RecursionError):
        return _reject(None, types.PARSE_ERROR, "Parse error")

    if isinstance(message, list):
        return _reject(None, types.INVALID_REQUEST, "Invalid Request", "Batch requests are not supported")
    if not isinstance(message, dict):
        return _reject(None, types.INVALID_REQUEST, "Invalid Request", "Message must be a JSON object")
    if "method" not in message and ("result" in message or "error" in message):
        return FORWARD

    problem = _envelope_problem(message)
    if problem is None:
        return FORWARD
    request_id = message.get("id")
    return _reject(
        request_id if _is_valid_id(request_id) else None, types.INVALID_REQUEST, "Invalid Request", problem
    )


def _envelope_problem(message: Mapping[str, Any]) -> str | None:
    if message.get("jsonrpc") != "2.0":
        return '"jsonrpc" must be exactly "2.0"'
    if "method" not in message:
        return '"method" is required'
    if not isinstance(message["method"], str):
        return '"method" must be a string'
    if "id" in message and not _is_valid_id(message["id"]):
        return '"id" must be a string or an integer'
    if "params" in message and not isinstance(message["params"], dict):
        return '"params" must be an object'
    return None


def _is_valid_id(value: object) -> bool:
    # bool is a subclass of int in Python, but `true` is not a valid JSON-RPC id.
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def _reject_non_standard_constant(name: str) -> None:
    raise ValueError(f"{name} is not valid JSON")


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

    Well-formed frames are yielded to the SDK's reader; malformed ones are answered directly on the
    outbound stream and never reach the dispatcher.
    """

    def __init__(self, source: anyio.AsyncFile[str]) -> None:
        self._source = source
        self._replies: ObjectSendStream[SessionMessage] | None = None
        self._bound = anyio.Event()

    def bind_replies(self, replies: ObjectSendStream[SessionMessage]) -> None:
        self._replies = replies
        self._bound.set()

    def __aiter__(self) -> CheckedLines:
        return self

    async def __anext__(self) -> str:
        while True:
            line = await self._source.readline()
            if not line:
                raise StopAsyncIteration
            check = check_frame(line)
            if check.forward:
                return line
            if check.error is not None:
                await self._answer(check.error)

    async def _answer(self, reply: types.JSONRPCError) -> None:
        # The data string is a fixed description, never a copy of the client's input.
        logger.warning("rejected malformed frame: %s (%s)", reply.error.message, reply.error.data or "-")
        # The SDK starts reading before it hands out the outbound stream.
        await self._bound.wait()
        assert self._replies is not None
        with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
            await self._replies.send(SessionMessage(reply))


# ---------------------------------------------------------------------------------------------
# stdin claim
# ---------------------------------------------------------------------------------------------
class _UnownedTextWrapper(io.TextIOWrapper):
    """Text layer that never closes the buffer under it (it may be the process's own stdin)."""

    def close(self) -> None:
        with contextlib.suppress(ValueError):
            self.detach()


@contextlib.contextmanager
def _claim_stdin() -> Iterator[IO[bytes]]:
    """Read the wire from a private duplicate of fd 0 and point fd 0 at the null device.

    Best effort, like the SDK's claim of fd 1: when stdin is not fd 0 (an embedded or test run) or
    the descriptors cannot be moved, the stream is read in place.
    """
    try:
        buffer: IO[bytes] = sys.stdin.buffer
        backed_by_fd0 = buffer.fileno() == 0
    except (AttributeError, OSError, ValueError):
        backed_by_fd0 = False
    if not backed_by_fd0:
        yield sys.stdin.buffer
        return

    try:
        private_fd = os.dup(0)
    except OSError:
        yield buffer
        return
    if private_fd <= 2:  # a standard fd was closed; never serve the wire from one
        os.close(private_fd)
        yield buffer
        return

    diverted = _point_fd_at(0, os.devnull)
    try:
        # closefd=False: after an interrupted shutdown a worker thread can still be blocked reading
        # this descriptor, so it is never closed (and so never recycled under that thread).
        yield os.fdopen(private_fd, "rb", closefd=False)
    finally:
        if diverted:
            with contextlib.suppress(OSError):
                os.dup2(private_fd, 0)
                rebind_std_handle_to_fd(0)


def _point_fd_at(fd: int, path: str) -> bool:
    try:
        target = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.dup2(target, fd)
    except OSError:
        return False
    finally:
        os.close(target)
    # os.dup2 only updates the C runtime table. On Windows, child processes inherit the Win32
    # standard-handle slot, so point that at the null device as well.
    with contextlib.suppress(OSError):
        rebind_std_handle_to_fd(fd)
    return True
