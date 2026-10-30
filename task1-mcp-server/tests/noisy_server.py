"""A server that misbehaves on purpose, for the stdout-isolation tests.

It writes to stdout in each way real code accidentally does, in every phase of the process's life.
Every marker below must end up on stderr and none on the wire:

* before `main()`: a buffered `print()` and an unflushed write to `sys.__stdout__`. Skipped when
  PYTHONUNBUFFERED is set: unbuffered output written before `main()` runs is on the wire before
  any server code exists, which no server can take back.
* startup, inside `create_server()`: `print()`, and a "library" logger whose `StreamHandler` was
  bound to the original `sys.stdout` at import time, as many libraries do
* while serving, on every customer lookup: `print()`, `os.write(1, ...)`, an unflushed write to
  `sys.__stdout__`, the library logger, and a child process that writes to stdout and reads stdin
* at exit, from an `atexit` hook: `print()`, `os.write(1, ...)` and `sys.__stdout__`
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys

from customer_mcp.__main__ import main
from customer_mcp.store import Customer, CustomerStore

CHILD = "import sys; print('child stdout'); print('child stdin:', repr(sys.stdin.read()))"

library_log = logging.getLogger("noisy.library")
library_log.addHandler(logging.StreamHandler(sys.stdout))  # the original stdout, captured at import
library_log.propagate = False
library_log.setLevel(logging.INFO)

_original_get = CustomerStore.get
_original_seed = CustomerStore.with_seed_data.__func__  # type: ignore[attr-defined]


def _noisy_seed(cls: type[CustomerStore]) -> CustomerStore:
    print("startup print()")
    library_log.info("startup library log")
    return _original_seed(cls)


def _noisy_get(self: CustomerStore, customer_id: str) -> Customer:
    print("stray print() from a tool handler")
    os.write(1, b"raw write to fd 1\n")
    sys.__stdout__.write("unflushed sys.__stdout__ write while serving\n")  # type: ignore[union-attr]
    library_log.info("library log while serving")
    subprocess.run([sys.executable, "-c", CHILD], check=True, timeout=30)
    return _original_get(self, customer_id)


def _noisy_exit() -> None:
    print("atexit print()")
    os.write(1, b"atexit write to fd 1\n")
    sys.__stdout__.write("atexit sys.__stdout__ write\n")  # type: ignore[union-attr]


if __name__ == "__main__":
    CustomerStore.get = _noisy_get  # type: ignore[method-assign]
    CustomerStore.with_seed_data = classmethod(_noisy_seed)  # type: ignore[method-assign, assignment]
    atexit.register(_noisy_exit)
    if not os.environ.get("PYTHONUNBUFFERED"):
        print("pre-main print()")
        sys.__stdout__.write("pre-main sys.__stdout__ write\n")  # type: ignore[union-attr]
    sys.exit(main())
