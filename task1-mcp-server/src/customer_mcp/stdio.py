"""Hardened stdio transport.

Over stdio, stdout *is* the JSON-RPC wire: one message per line, nothing else. A single stray
`print()` corrupts the stream and the client drops the connection. Three layers keep it clean:

1. The SDK's `stdio_server()` claims fd 1. It serves the wire from a private duplicate and
   points fd 1 at stderr, so output from C extensions, `os.write(1, ...)` and child processes
   lands on stderr.
2. `sys.stdout` is rebound to `sys.stderr` while serving. `print()` never touches the wire,
   even on platforms where the fd claim falls back to sharing the original stdout buffer.
3. Logging is configured to stderr only (see `logging_config`).
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server


async def serve_stdio(server: Server[Any]) -> None:
    """Serve `server` over stdin/stdout until the client closes stdin."""
    # Order matters: the SDK must claim the *real* stdout before we rebind sys.stdout.
    async with stdio_server() as (read_stream, write_stream):
        with contextlib.redirect_stdout(sys.stderr):
            await server.run(read_stream, write_stream, server.create_initialization_options())
