"""A server for the Ctrl+C tests: SIGINT arrives while the client still holds stdin open.

With `--simulate`, the first customer lookup arms a timer that delivers SIGINT through
`_thread.interrupt_main()`, half a second later, once the reply has gone out and the server is idle
waiting for stdin. That runs Python's SIGINT handling exactly as a real Ctrl+C does, and works on
every platform; without it, the test sends a real SIGINT (POSIX only).
"""

from __future__ import annotations

import _thread
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
    _thread.interrupt_main()


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
