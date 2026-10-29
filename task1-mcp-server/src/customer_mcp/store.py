"""In-memory customer store and refund ledger.

Stands in for a CRM / payments backend. Money is handled as `Decimal` end to end so refund
arithmetic never suffers from binary floating-point rounding.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

import anyio


class CustomerStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Customer:
    customer_id: str
    name: str
    email: str
    tier: str
    status: CustomerStatus
    currency: str
    lifetime_spend: Decimal
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Refund:
    refund_id: str
    customer_id: str
    amount: Decimal
    currency: str
    reason: str
    created_at: datetime


class CustomerNotFoundError(LookupError):
    def __init__(self, customer_id: str) -> None:
        super().__init__(f"Customer {customer_id} was not found")
        self.customer_id = customer_id


class RefundRejectedError(Exception):
    """A refund request that is well-formed but violates a business rule."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class CustomerStore:
    """Thread-unsafe, task-safe store: mutations are serialised by an `anyio.Lock`."""

    def __init__(self, customers: Iterable[Customer]) -> None:
        self._customers: dict[str, Customer] = {c.customer_id: c for c in customers}
        self._refunds: dict[str, list[Refund]] = {}
        self._lock = anyio.Lock()

    def get(self, customer_id: str) -> Customer:
        try:
            return self._customers[customer_id]
        except KeyError:
            raise CustomerNotFoundError(customer_id) from None

    def refunds_for(self, customer_id: str) -> tuple[Refund, ...]:
        return tuple(self._refunds.get(customer_id, ()))

    def refundable_balance(self, customer_id: str) -> Decimal:
        customer = self.get(customer_id)
        refunded = sum((r.amount for r in self._refunds.get(customer_id, ())), Decimal("0"))
        return customer.lifetime_spend - refunded

    async def create_refund(self, customer_id: str, amount: Decimal, reason: str) -> Refund:
        """Record a refund after checking account status and refundable balance atomically."""
        async with self._lock:
            customer = self.get(customer_id)
            if customer.status is not CustomerStatus.ACTIVE:
                raise RefundRejectedError(
                    "customer_not_active",
                    f"Refunds are not allowed for customers with status '{customer.status}'",
                )
            balance = self.refundable_balance(customer_id)
            if amount > balance:
                raise RefundRejectedError(
                    "exceeds_refundable_balance",
                    f"Refund of {amount} {customer.currency} exceeds the refundable balance of "
                    f"{balance} {customer.currency}",
                )
            refund = Refund(
                refund_id=f"RF-{uuid.uuid4().hex[:12].upper()}",
                customer_id=customer_id,
                amount=amount,
                currency=customer.currency,
                reason=reason,
                created_at=datetime.now(UTC),
            )
            self._refunds.setdefault(customer_id, []).append(refund)
            return refund

    @classmethod
    def with_seed_data(cls) -> CustomerStore:
        def ts(value: str) -> datetime:
            return datetime.fromisoformat(value).replace(tzinfo=UTC)

        return cls(
            [
                Customer(
                    "CUST-10001",
                    "Ada Lovelace",
                    "ada@example.com",
                    "gold",
                    CustomerStatus.ACTIVE,
                    "USD",
                    Decimal("1250.00"),
                    ts("2023-02-14T09:30:00"),
                ),
                Customer(
                    "CUST-10002",
                    "Alan Turing",
                    "alan.turing@example.org",
                    "silver",
                    CustomerStatus.ACTIVE,
                    "USD",
                    Decimal("310.50"),
                    ts("2023-06-23T16:05:00"),
                ),
                Customer(
                    "CUST-10003",
                    "Grace Hopper",
                    "grace@example.net",
                    "platinum",
                    CustomerStatus.ACTIVE,
                    "USD",
                    Decimal("9800.00"),
                    ts("2022-12-09T11:00:00"),
                ),
                Customer(
                    "CUST-10004",
                    "Edsger Dijkstra",
                    "edsger@example.com",
                    "standard",
                    CustomerStatus.SUSPENDED,
                    "EUR",
                    Decimal("45.00"),
                    ts("2024-05-11T08:15:00"),
                ),
                Customer(
                    "CUST-10005",
                    "Barbara Liskov",
                    "barbara@example.org",
                    "standard",
                    CustomerStatus.CLOSED,
                    "USD",
                    Decimal("0.00"),
                    ts("2021-11-07T13:45:00"),
                ),
            ]
        )
