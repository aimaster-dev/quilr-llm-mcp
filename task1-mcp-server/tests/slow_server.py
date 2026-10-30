"""A server whose refunds never finish, for the shutdown tests.

`trigger_refund` blocks forever, so a refund request stays in flight until the client cancels it or
the server gives up waiting for it at EOF. argv[1], if given, overrides that wait in seconds.
"""

from __future__ import annotations

import sys
from decimal import Decimal

import anyio

from customer_mcp import stdio
from customer_mcp.__main__ import main
from customer_mcp.store import CustomerStore, Refund


async def _stuck_refund(self: CustomerStore, customer_id: str, amount: Decimal, reason: str) -> Refund:
    await anyio.sleep_forever()
    raise AssertionError("unreachable")


if __name__ == "__main__":
    CustomerStore.create_refund = _stuck_refund  # type: ignore[method-assign]
    if len(sys.argv) > 1:
        stdio.EOF_DRAIN_TIMEOUT = float(sys.argv.pop(1))
    main()
