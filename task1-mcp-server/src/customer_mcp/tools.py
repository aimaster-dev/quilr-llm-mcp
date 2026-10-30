"""The customer-service tools: `get_customer_record` and `trigger_refund`."""

from __future__ import annotations

import logging
from decimal import Decimal

import mcp_types as types

from customer_mcp.registry import ToolExecutionError, ToolSpec
from customer_mcp.schemas import (
    CustomerRecord,
    GetCustomerRecordInput,
    RefundReceipt,
    TriggerRefundInput,
)
from customer_mcp.store import CustomerNotFoundError, CustomerStore, RefundRejectedError

logger = logging.getLogger(__name__)

CENT = Decimal("0.01")


def _money(value: Decimal) -> str:
    return f"{value.quantize(CENT)}"


def customer_tools(store: CustomerStore) -> list[ToolSpec]:
    async def get_customer_record(params: GetCustomerRecordInput) -> CustomerRecord:
        try:
            customer = store.get(params.customer_id)
        except CustomerNotFoundError as exc:
            raise ToolExecutionError("customer_not_found", str(exc)) from None
        return CustomerRecord(
            customer_id=customer.customer_id,
            name=customer.name,
            email=customer.email,
            tier=customer.tier,
            status=customer.status.value,
            currency=customer.currency,
            lifetime_spend=_money(customer.lifetime_spend),
            refundable_balance=_money(store.refundable_balance(customer.customer_id)),
            created_at=customer.created_at,
        )

    async def trigger_refund(params: TriggerRefundInput) -> RefundReceipt:
        # Validation guarantees <= 2 decimal places, so this conversion is exact.
        amount = Decimal(repr(params.amount)).quantize(CENT)
        try:
            refund = await store.create_refund(params.customer_id, amount, params.reason)
        except CustomerNotFoundError as exc:
            raise ToolExecutionError("customer_not_found", str(exc)) from None
        except RefundRejectedError as exc:
            raise ToolExecutionError(exc.code, exc.message) from None

        # The free-text reason is deliberately not logged: it may contain customer PII.
        logger.info("refund %s issued: customer=%s amount=%s", refund.refund_id, refund.customer_id, amount)
        return RefundReceipt(
            refund_id=refund.refund_id,
            customer_id=refund.customer_id,
            amount=_money(refund.amount),
            currency=refund.currency,
            reason=refund.reason,
            status="processed",
            remaining_refundable_balance=_money(store.refundable_balance(refund.customer_id)),
            created_at=refund.created_at,
        )

    return [
        ToolSpec(
            name="get_customer_record",
            title="Get customer record",
            description=(
                "Fetch a customer's profile, account status and refundable balance by customer ID "
                "(format CUST-XXXXX, five digits)."
            ),
            input_model=GetCustomerRecordInput,
            output_model=CustomerRecord,
            handler=get_customer_record,
            annotations=types.ToolAnnotations(
                read_only_hint=True, idempotent_hint=True, open_world_hint=False
            ),
        ),
        ToolSpec(
            name="trigger_refund",
            title="Trigger refund",
            description=(
                "Issue a refund to a customer. Irreversible: the amount must not exceed the customer's "
                "refundable balance and the account must be active. Requires a reason of at least "
                "10 characters."
            ),
            input_model=TriggerRefundInput,
            output_model=RefundReceipt,
            handler=trigger_refund,
            # Closed world, like the lookup: both tools touch only the server's own store. Backed
            # by a real CRM or payments provider, both would become open-world.
            annotations=types.ToolAnnotations(
                read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False
            ),
        ),
    ]
