"""Input and output contracts for the customer tools.

Inputs are validated in Pydantic **strict mode** with unknown fields forbidden:

* no type coercion: `"25.00"` is not a number, `true` is not a number, `10001` is not a string
* `extra="forbid"` so typos (`customerId`) and smuggled fields fail loudly instead of being ignored
* the same models generate the JSON Schema published in `tools/list`. Where a rule is enforced by
  a custom validator, the schema states it with an equivalent (or stricter) JSON Schema construct;
  the README lists the few rules JSON Schema cannot express.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints
from pydantic_core import PydanticCustomError

# `[0-9]`, not `\d`: Pydantic's regex engine treats `\d` as Unicode-aware, which would accept
# look-alike digits such as "CUST-١٢٣٤٥" (Arabic-Indic). `$` is end-of-text (no trailing "\n").
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
CUSTOMER_ID_LENGTH = 10
MAX_REFUND_AMOUNT = 1_000_000
REASON_MIN_LENGTH = 10
REASON_MAX_LENGTH = 500
_ALLOWED_CONTROL_CHARS = frozenset("\t\n\r")
# Zero-width non-joiner and joiner: invisible, but Persian and Indic orthography and emoji
# sequences (👨‍👩‍👧) need them. They are allowed only between two visible characters.
_JOINERS = frozenset("‌‍")


def _at_most_two_decimal_places(value: float) -> float:
    # repr() gives the shortest round-tripping literal: 19.99 -> "19.99", 0.1 + 0.2 -> "0.30000000000000004".
    exponent = Decimal(repr(value)).as_tuple().exponent
    if isinstance(exponent, int) and exponent < -2:
        raise PydanticCustomError("decimal_places", "Amount must have at most 2 decimal places")
    return value


def _nfc(value: object) -> object:
    # Canonically equivalent spellings ("é" vs "e" + U+0301) are stored, and counted, the same way.
    return unicodedata.normalize("NFC", value) if isinstance(value, str) else value


def _is_forbidden(ch: str) -> bool:
    category = unicodedata.category(ch)
    return (category == "Cc" and ch not in _ALLOWED_CONTROL_CHARS) or (
        category == "Cf" and ch not in _JOINERS
    )


def _is_visible(ch: str) -> bool:
    return not ch.isspace() and unicodedata.category(ch) not in ("Cc", "Cf")


def _no_control_characters(value: str) -> str:
    if any(unicodedata.category(ch) == "Cc" and ch not in _ALLOWED_CONTROL_CHARS for ch in value):
        raise PydanticCustomError("control_characters", "Value must not contain control characters")
    return value


def _no_format_characters(value: str) -> str:
    """Reject invisible format characters (Unicode category Cf).

    Zero-width spaces and BOMs let a reason meet the length minimum while showing nothing; bidi
    overrides make the stored text read differently from what it contains; tag characters can
    smuggle hidden instructions to a model that reads the reason back. ZWJ/ZWNJ are the exception,
    in joining position only, so they cannot pad a reason either.
    """
    last = len(value) - 1
    for index, ch in enumerate(value):
        if unicodedata.category(ch) != "Cf":
            continue
        if (
            ch in _JOINERS
            and 0 < index < last
            and _is_visible(value[index - 1])
            and _is_visible(value[index + 1])
        ):
            continue
        raise PydanticCustomError(
            "format_characters",
            "Value must not contain invisible format characters (zero-width, bidi or tag characters)",
        )
    return value


def _character_class(chars: Iterable[str]) -> str:
    """The body of a regex character class matching `chars`, as `\\uXXXX` ranges.

    `\\uXXXX` escapes mean the same in ECMA-262 (the dialect JSON Schema specifies) and in Python's
    `re`, so the published pattern behaves the same in a JS host and in `jsonschema`.
    """
    codes = sorted({ord(ch) for ch in chars})
    ranges: list[list[int]] = []
    for code in codes:
        if ranges and code == ranges[-1][1] + 1:
            ranges[-1][1] = code
        else:
            ranges.append([code, code])
    return "".join(f"\\u{lo:04X}" if lo == hi else f"\\u{lo:04X}-\\u{hi:04X}" for lo, hi in ranges)


_BMP = [chr(code) for code in range(0x10000) if not 0xD800 <= code <= 0xDFFF]
_FORBIDDEN_CLASS = _character_class(ch for ch in _BMP if _is_forbidden(ch))
_SPACE_CLASS = _character_class(ch for ch in _BMP if ch.isspace())
# Published for `reason`: no control or format characters anywhere, and no whitespace at either end.
# The server trims edge whitespace instead of rejecting it, so for every value this pattern accepts,
# trimming is a no-op and `minLength`/`maxLength` mean exactly what the server checks.
REASON_PATTERN = (
    f"^[^{_SPACE_CLASS}{_FORBIDDEN_CLASS}](?:[^{_FORBIDDEN_CLASS}]*[^{_SPACE_CLASS}{_FORBIDDEN_CLASS}])?$"
)


CustomerId = Annotated[
    str,
    Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier: 'CUST-' followed by exactly five ASCII digits.",
        examples=["CUST-10001"],
        # Implied by the anchored pattern; published as well because Python's `re` (used by the
        # `jsonschema` package) lets `$` match before a trailing newline, unlike ECMA-262.
        json_schema_extra={"minLength": CUSTOMER_ID_LENGTH, "maxLength": CUSTOMER_ID_LENGTH},
    ),
]

RefundAmount = Annotated[
    float,
    Field(
        gt=0,
        le=MAX_REFUND_AMOUNT,
        allow_inf_nan=False,
        description=(
            "Refund amount in the customer's account currency. Must be greater than 0, at most "
            "1,000,000, finite, and have at most 2 decimal places (checked by the server)."
        ),
        examples=[25.0, 199.99],
    ),
    AfterValidator(_at_most_two_decimal_places),
]

RefundReason = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=REASON_MIN_LENGTH, max_length=REASON_MAX_LENGTH),
    # Listed after the constraints so it wraps them: normalization happens before trim and length.
    BeforeValidator(_nfc),
    AfterValidator(_no_control_characters),
    AfterValidator(_no_format_characters),
    Field(
        description=(
            f"Why the refund is issued: {REASON_MIN_LENGTH}-{REASON_MAX_LENGTH} characters (Unicode code "
            "points, after NFC normalization and trimming leading/trailing whitespace). Line breaks and "
            "tabs are allowed; other control characters and invisible format characters are not."
        ),
        examples=["Item arrived damaged in transit"],
        json_schema_extra={"pattern": REASON_PATTERN},
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


# ---------------------------------------------------------------------------------------------
# Tool execution errors (`isError: true`). The `outputSchema` describes successful results only,
# so an error result carries no `structuredContent`: its text content is this JSON document.
# ---------------------------------------------------------------------------------------------
class ToolErrorDetail(BaseModel):
    code: str = Field(
        description="Machine-readable reason the call failed.",
        examples=["customer_not_found", "customer_not_active", "exceeds_refundable_balance"],
    )
    message: str = Field(description="Human-readable explanation.")


class ToolErrorText(BaseModel):
    """The text content of an `isError: true` tool result, as JSON."""

    error: ToolErrorDetail
