"""Input and output contracts for the customer tools.

Inputs are validated in Pydantic **strict mode** with unknown fields forbidden:

* no type coercion: `"25.00"` is not a number, `true` is not a number, `10001` is not a string
* `extra="forbid"` so typos (`customerId`) and smuggled fields fail loudly instead of being ignored
* the same models generate the JSON Schema published in `tools/list`
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints
from pydantic_core import PydanticCustomError

# `[0-9]`, not `\d`: Pydantic's regex engine treats `\d` as Unicode-aware, which would accept
# look-alike digits such as "CUST-١٢٣٤٥" (Arabic-Indic). `$` is end-of-text (no trailing "\n").
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
MAX_REFUND_AMOUNT = 1_000_000
REASON_MIN_LENGTH = 10
REASON_MAX_LENGTH = 500
_ALLOWED_CONTROL_CHARS = frozenset("\t\n\r")


def _at_most_two_decimal_places(value: float) -> float:
    # repr() gives the shortest round-tripping literal: 19.99 -> "19.99", 0.1 + 0.2 -> "0.30000000000000004".
    exponent = Decimal(repr(value)).as_tuple().exponent
    if isinstance(exponent, int) and exponent < -2:
        raise PydanticCustomError("decimal_places", "Amount must have at most 2 decimal places")
    return value


def _no_control_characters(value: str) -> str:
    if any(unicodedata.category(ch) == "Cc" and ch not in _ALLOWED_CONTROL_CHARS for ch in value):
        raise PydanticCustomError("control_characters", "Value must not contain control characters")
    return value


CustomerId = Annotated[
    str,
    Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier: 'CUST-' followed by exactly five digits.",
        examples=["CUST-10001"],
    ),
]

RefundAmount = Annotated[
    float,
    Field(
        gt=0,
        le=MAX_REFUND_AMOUNT,
        allow_inf_nan=False,
        description=(
            "Refund amount in the customer's account currency. Must be greater than 0, "
            "finite, and have at most 2 decimal places."
        ),
        examples=[25.0, 199.99],
    ),
    AfterValidator(_at_most_two_decimal_places),
]

RefundReason = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=REASON_MIN_LENGTH, max_length=REASON_MAX_LENGTH),
    AfterValidator(_no_control_characters),
    Field(
        description=(
            f"Why the refund is issued ({REASON_MIN_LENGTH}-{REASON_MAX_LENGTH} characters after trimming "
            "leading/trailing whitespace)."
        ),
        examples=["Item arrived damaged in transit"],
    ),
]


class StrictInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class GetCustomerRecordInput(StrictInput):
    """Arguments for `get_customer_record`."""

    customer_id: CustomerId


class TriggerRefundInput(StrictInput):
    """Arguments for `trigger_refund`."""

    customer_id: CustomerId
    amount: RefundAmount
    reason: RefundReason


# ---------------------------------------------------------------------------------------------
# Outputs. Money is rendered as a fixed-point decimal string ("1250.00") to avoid float drift.
# ---------------------------------------------------------------------------------------------
MoneyString = Annotated[str, Field(pattern=r"^-?[0-9]+\.[0-9]{2}$", examples=["1250.00"])]


class CustomerRecord(BaseModel):
    customer_id: str
    name: str
    email: str
    tier: str
    status: Literal["active", "suspended", "closed"]
    currency: str
    lifetime_spend: MoneyString
    refundable_balance: MoneyString
    created_at: datetime


class RefundReceipt(BaseModel):
    refund_id: str
    customer_id: str
    amount: MoneyString
    currency: str
    reason: str
    status: Literal["processed"]
    remaining_refundable_balance: MoneyString
    created_at: datetime
