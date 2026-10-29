"""A server that misbehaves on purpose, for the stdout-isolation tests.

Every customer lookup writes to stdout in each way real code accidentally does: `print()`, a raw
`os.write` to fd 1, and a child process that inherits the standard streams (and tries to read stdin).
None of it may reach the JSON-RPC wire.
"""

from __future__ import annotations

import os
import subprocess
import sys

from customer_mcp.__main__ import main
from customer_mcp.store import Customer, CustomerStore

CHILD = "import sys; print('child stdout'); print('child stdin:', repr(sys.stdin.read()))"

_original_get = CustomerStore.get


def _noisy_get(self: CustomerStore, customer_id: str) -> Customer:
    print("stray print() from a tool handler")
    os.write(1, b"raw write to fd 1\n")
    subprocess.run([sys.executable, "-c", CHILD], check=True, timeout=30)
    return _original_get(self, customer_id)


if __name__ == "__main__":
    CustomerStore.get = _noisy_get  # type: ignore[method-assign]
    main()
