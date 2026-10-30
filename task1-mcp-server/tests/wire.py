"""Raw-pipe test harness: drives a server subprocess byte-for-byte, the way an MCP host does."""

from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

HANDSHAKE_VERSION = "2025-11-25"
SERVER_CMD = [sys.executable, "-m", "customer_mcp"]


def server_env(env: dict[str, str] | None = None) -> dict[str, str]:
    return {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}


class StdioWireClient:
    """Spawns a server and records every byte it writes to stdout and stderr."""

    def __init__(self, args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
        self.proc = subprocess.Popen(
            args or SERVER_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=server_env(env),
            cwd=Path(__file__).resolve().parent,
        )
        self.stdout_lines: list[bytes] = []
        self._stderr = bytearray()
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._ids = itertools.count(1)
        self._threads = [
            threading.Thread(target=self._pump_stdout, daemon=True),
            threading.Thread(target=self._pump_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.stdout_lines.append(line)
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for chunk in iter(lambda: self.proc.stderr.read1(4096), b""):  # type: ignore[union-attr]
            self._stderr.extend(chunk)

    # -- sending -------------------------------------------------------------------------
    def send_raw(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def send(self, message: Any) -> None:
        self.send_raw(json.dumps(message).encode() + b"\n")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = next(self._ids)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        response = self.receive()
        assert response.get("id") == request_id, response
        return response

    # -- receiving -----------------------------------------------------------------------
    def receive(self, timeout: float = 15.0) -> dict[str, Any]:
        line = self._lines.get(timeout=timeout)
        if line is None:
            raise EOFError(f"server closed stdout; stderr:\n{self.stderr}")
        return json.loads(line)

    @property
    def stderr(self) -> str:
        return self._stderr.decode("utf-8", errors="replace")

    # -- lifecycle -----------------------------------------------------------------------
    def initialize(self) -> dict[str, Any]:
        response = self.request(
            "initialize",
            {
                "protocolVersion": HANDSHAKE_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "wire-test", "version": "1.0"},
            },
        )
        self.notify("notifications/initialized")
        return response

    def close(self) -> int:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            code = self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait()
        for thread in self._threads:
            thread.join(timeout=5)
        return code

    def __enter__(self) -> StdioWireClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def assert_pure_jsonrpc(lines: list[bytes]) -> None:
    """Every byte the server wrote to stdout must be a newline-delimited JSON-RPC 2.0 message."""
    assert lines, "server wrote nothing to stdout"
    for raw in lines:
        assert raw.endswith(b"\n"), f"unterminated stdout line: {raw!r}"
        message = json.loads(raw)
        assert isinstance(message, dict), raw
        assert message.get("jsonrpc") == "2.0", raw
        assert ("id" in message) or ("method" in message), raw
