"""A server for the Ctrl+C tests: SIGINT arrives while the client still holds stdin open.

With `--simulate`, the first customer lookup arms a timer that interrupts the server from inside,
half a second later, once the reply has gone out and the server is idle waiting for stdin. Without
it, the test sends a real SIGINT from outside (POSIX only).

How the interrupt is delivered matters. An idle event loop is blocked in the OS (`epoll_wait`,
`select`), and Python only runs its SIGINT handling once that call returns:

* POSIX: `_thread.interrupt_main()` only sets the interpreter's flag and does not interrupt the
  blocked call, so the loop would sleep on until something else woke it. A real process-directed
  SIGINT (`os.kill(os.getpid(), ...)`, which Linux delivers to the main thread) interrupts it, exactly
  like Ctrl+C in a terminal.
* Windows: `os.kill` cannot send SIGINT to a single process, and `_thread.interrupt_main()` also
  signals the event that the loop's wait includes, so it wakes the loop as Ctrl+C does.
"""

from __future__ import annotations

import _thread
import os
import signal
import sys
import threading
import time

from customer_mcp.__main__ import main
from customer_mcp.store import Customer, CustomerStore

_original_get = CustomerStore.get
_armed = threading.Event()


def _interrupt_soon() -> None:
    time.sleep(0.5)
    if sys.platform == "win32":
        _thread.interrupt_main()
    else:
        os.kill(os.getpid(), signal.SIGINT)


def _get_then_interrupt(self: CustomerStore, customer_id: str) -> Customer:
    if not _armed.is_set():
        _armed.set()
        threading.Thread(target=_interrupt_soon, daemon=True).start()
    return _original_get(self, customer_id)


if __name__ == "__main__":
    # A process can inherit SIGINT as ignored (from a non-interactive shell, say), and then Python
    # never turns it into KeyboardInterrupt. Restore the default handling the test relies on.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if "--simulate" in sys.argv:
        sys.argv.remove("--simulate")
        CustomerStore.get = _get_then_interrupt  # type: ignore[method-assign]
    sys.exit(main())
