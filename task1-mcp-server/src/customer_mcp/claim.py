"""Claim stdin and stdout for the JSON-RPC wire, once, for the lifetime of the process.

Over stdio, stdout *is* the wire: one JSON-RPC message per line and nothing else. Any other byte
that reaches it (a `print()`, a library log handler, a C extension, a child process) corrupts the
stream. `claim_stdio()` runs as the very first action of `main()` and never undoes its work:

* The wire is served from private duplicates of fd 0 and fd 1 (non-inheritable, above the standard
  range), so only the server reads requests and writes replies.
* fd 1 is pointed at stderr and fd 0 at the null device. Output from `os.write(1, ...)`, C code
  and child processes lands on stderr; a library or child that reads stdin gets EOF. On Windows
  the Win32 standard handles are repointed too, because child processes inherit those.
* `sys.stdout` is rebound to `sys.stderr`, and whatever was still buffered in the original
  `sys.stdout` is flushed *after* fd 1 was diverted, so it lands on stderr as well.

fd 1 is never pointed back at the wire, not even at exit. The SDK's own `stdio_server()` restores
it when serving ends, after which output buffered in `sys.__stdout__` or written by an `atexit`
hook would be flushed onto the wire behind the last JSON-RPC message.

This module imports only the standard library at import time, so `__main__` can claim the wire
before importing the SDK and the server. Output written to stdout *before* `main()` runs is beyond
its reach unless it is still sitting in Python's stdout buffer.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO

if sys.platform != "win32":
    import fcntl


class WireUnavailableError(OSError):
    """stdout is not open, so there is nowhere to send replies."""


@dataclass
class StdioWire:
    """The claimed wire: where the server reads requests and writes replies."""

    reader: IO[bytes]
    writer: IO[bytes]
    # Problems found while claiming, reported once logging is configured.
    notes: list[str] = field(default_factory=list)


def claim_stdio() -> StdioWire:
    """Take the process's stdout and stdin for the wire. Call once, before anything else runs.

    Raises:
        WireUnavailableError: fd 1 is closed.
    """
    notes: list[str] = []
    writer = _claim_stdout(notes)  # first: from here on nothing else can write to the wire
    reader = _claim_stdin(notes)
    return StdioWire(reader=reader, writer=writer, notes=notes)


# ---------------------------------------------------------------------------------------------
# stdout
# ---------------------------------------------------------------------------------------------
def _claim_stdout(notes: list[str]) -> IO[bytes]:
    try:
        wire_fd = _dup_above_std(1)
        writer = _open_fd(wire_fd, "wb")
    except OSError as exc:
        # fd 1 is closed, or (`customer-mcp >&-` from a Windows shell) holds an invalid handle.
        _rebind_sys_stdout()
        raise WireUnavailableError("stdout is not open: there is no JSON-RPC wire to serve on") from exc

    if not _divert(1, _open_stdout_diversion, wire_fd):
        # fd 1 still carries the wire, so serve it in place. print() is still kept off it.
        writer.close()
        _rebind_sys_stdout()
        notes.append("could not divert fd 1; stray writes to fd 1 can still reach the wire")
        return os.fdopen(1, "wb", closefd=False)

    _rebind_sys_stdout()
    _rebind_win32_handle(1)
    return writer


def _rebind_sys_stdout() -> None:
    """Point `sys.stdout` at stderr and flush whatever was buffered for stdout before the claim.

    Called once fd 1 is diverted, so the flushed bytes land on stderr instead of reaching the wire
    at exit. The original object stays in `sys.__stdout__`; it writes to fd 1, which never points
    at the wire again.
    """
    previous = sys.stdout
    sys.stdout = sys.stderr
    for stream in {id(s): s for s in (previous, sys.__stdout__) if s is not None}.values():
        with contextlib.suppress(Exception):  # a closed or broken stream must not stop the server
            stream.flush()


def _open_stdout_diversion() -> int:
    """stderr if it is usable, else the null device."""
    try:
        stderr = os.dup(2)
    except OSError:
        return os.open(os.devnull, os.O_WRONLY)
    if _is_valid(stderr):
        return stderr
    os.close(stderr)
    return os.open(os.devnull, os.O_WRONLY)


# ---------------------------------------------------------------------------------------------
# stdin
# ---------------------------------------------------------------------------------------------
def _claim_stdin(notes: list[str]) -> IO[bytes]:
    try:
        wire_fd = _dup_above_std(0)
        reader = _open_fd(wire_fd, "rb")
    except OSError:
        # fd 0 is closed or holds an invalid handle (`customer-mcp <&-`, or no console under
        # pythonw): read EOF at once. Fill the slot with the null device, so no file opened later
        # lands on fd 0 for children to read.
        notes.append("stdin is not open; serving an empty session")
        with contextlib.suppress(OSError):
            null = _open_stdin_diversion()
            if null != 0:
                os.dup2(null, 0)
                os.close(null)
            _rebind_win32_handle(0)
        return io.BytesIO()

    if not _divert(0, _open_stdin_diversion, wire_fd):
        # fd 0 still carries the wire: read it in place rather than through a second reader.
        reader.close()
        notes.append("could not divert fd 0; a child process could consume requests")
        return os.fdopen(0, "rb", closefd=False)

    _rebind_win32_handle(0)
    return reader


def _open_stdin_diversion() -> int:
    return os.open(os.devnull, os.O_RDONLY)


# ---------------------------------------------------------------------------------------------
# Descriptor plumbing
# ---------------------------------------------------------------------------------------------
def _dup_above_std(fd: int) -> int:
    """A non-inheritable duplicate of `fd` that cannot land on 0, 1 or 2.

    If a standard descriptor is closed, a plain `dup` would reuse it, and the wire would end up on
    what other code believes is stdin or stderr.
    """
    if sys.platform != "win32":
        return fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)
    held: list[int] = []
    try:
        duplicate = os.dup(fd)  # os.dup creates non-inheritable descriptors
        while duplicate <= 2:
            held.append(duplicate)
            duplicate = os.dup(fd)
        return duplicate
    finally:
        for low in held:
            os.close(low)


def _open_fd(fd: int, mode: str) -> IO[bytes]:
    """`os.fdopen`, closing `fd` if it fails (it checks the descriptor, unlike `os.dup`)."""
    try:
        return os.fdopen(fd, mode)
    except OSError:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


def _is_valid(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def _divert(fd: int, open_diversion: Callable[[], int], wire_fd: int) -> bool:
    """Point `fd` at a diversion target; on failure make sure it still carries the wire."""
    try:
        target = open_diversion()
    except OSError:
        return False
    try:
        os.dup2(target, fd)
    except OSError:
        # A failed dup2 on Windows can close its target first: put the wire back on fd.
        with contextlib.suppress(OSError):
            os.dup2(wire_fd, fd)
        return False
    finally:
        with contextlib.suppress(OSError):
            os.close(target)
    return True


def _rebind_win32_handle(fd: int) -> None:
    """`os.dup2` only updates the C runtime's table; children inherit the Win32 standard handles."""
    if sys.platform != "win32":
        return
    # Imported here, after the diversion: importing the SDK must not happen before the claim.
    from mcp.os.win32.utilities import rebind_std_handle_to_fd

    with contextlib.suppress(OSError):
        rebind_std_handle_to_fd(fd)
