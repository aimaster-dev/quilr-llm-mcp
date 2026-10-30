"""Edge cases for the strict input contracts."""

from __future__ import annotations

import math
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from customer_mcp.schemas import REASON_PATTERN, GetCustomerRecordInput, TriggerRefundInput

VALID_REFUND: dict[str, Any] = {
    "customer_id": "CUST-10001",
    "amount": 25.5,
    "reason": "Item arrived damaged in transit",
}


def error_types(exc: pytest.ExceptionInfo[ValidationError]) -> set[str]:
    return {e["type"] for e in exc.value.errors()}


class TestCustomerId:
    @pytest.mark.parametrize("value", ["CUST-00000", "CUST-10001", "CUST-99999"])
    def test_accepts_canonical_format(self, value: str) -> None:
        assert GetCustomerRecordInput.model_validate({"customer_id": value}).customer_id == value

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "CUST-1234",  # too few digits
            "CUST-123456",  # too many digits
            "cust-12345",  # wrong case
            "CUST_12345",  # wrong separator
            "CUST-12a45",  # non-digit
            " CUST-12345",  # leading whitespace
            "CUST-12345 ",  # trailing whitespace
            "CUST-12345\n",  # a trailing newline must not satisfy `$`
            "CUST-" + "".join(chr(0x0661 + i) for i in range(5)),  # Arabic-Indic digits 1-5
            "CUST-" + "".join(chr(0xFF11 + i) for i in range(5)),  # full-width digits 1-5
            "XCUST-12345",
        ],
    )
    def test_rejects_malformed_ids(self, value: str) -> None:
        with pytest.raises(ValidationError) as exc:
            GetCustomerRecordInput.model_validate({"customer_id": value})
        assert error_types(exc) == {"string_pattern_mismatch"}

    @pytest.mark.parametrize("value", [12345, None, ["CUST-12345"], {"id": "CUST-12345"}, True])
    def test_rejects_non_strings_without_coercion(self, value: Any) -> None:
        with pytest.raises(ValidationError) as exc:
            GetCustomerRecordInput.model_validate({"customer_id": value})
        assert error_types(exc) == {"string_type"}

    def test_missing_field(self) -> None:
        with pytest.raises(ValidationError) as exc:
            GetCustomerRecordInput.model_validate({})
        assert error_types(exc) == {"missing"}

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError) as exc:
            GetCustomerRecordInput.model_validate({"customer_id": "CUST-12345", "customerId": "x"})
        assert error_types(exc) == {"extra_forbidden"}


class TestRefundAmount:
    @pytest.mark.parametrize("amount", [0.01, 1, 25.5, 199.99, 1_000_000])
    def test_accepts_positive_amounts(self, amount: float) -> None:
        model = TriggerRefundInput.model_validate({**VALID_REFUND, "amount": amount})
        assert model.amount == amount
        assert isinstance(model.amount, float)

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (0, "greater_than"),
            (0.0, "greater_than"),
            (-0.0, "greater_than"),
            (-5, "greater_than"),
            (1_000_000.01, "less_than_equal"),
            (math.inf, "finite_number"),
            (-math.inf, "finite_number"),
            (math.nan, "finite_number"),
            (10.555, "decimal_places"),
            (0.001, "decimal_places"),
            (0.1 + 0.2, "decimal_places"),
        ],
    )
    def test_rejects_out_of_range_amounts(self, amount: float, expected: str) -> None:
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "amount": amount})
        assert error_types(exc) == {expected}

    @pytest.mark.parametrize("amount", ["25.00", "1e3", True, False, None, [25], {"value": 25}])
    def test_rejects_non_numbers_without_coercion(self, amount: Any) -> None:
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "amount": amount})
        assert error_types(exc) == {"float_type"}


class TestRefundReason:
    def test_minimum_length_is_ten_characters(self) -> None:
        assert TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "0123456789"})
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "012345678"})
        assert error_types(exc) == {"string_too_short"}

    def test_whitespace_padding_does_not_satisfy_minimum(self) -> None:
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "   damaged     "})
        assert error_types(exc) == {"string_too_short"}

    def test_whitespace_only_reason_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": " " * 50})

    def test_reason_is_trimmed(self) -> None:
        model = TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "  Item arrived damaged  "})
        assert model.reason == "Item arrived damaged"

    def test_maximum_length(self) -> None:
        assert TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "x" * 500})
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "x" * 501})
        assert error_types(exc) == {"string_too_long"}

    @pytest.mark.parametrize("reason", ["damaged item\x00 returned", "damaged\x1b[31m item returned"])
    def test_rejects_control_characters(self, reason: str) -> None:
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason})
        assert error_types(exc) == {"control_characters"}

    def test_allows_newlines_and_tabs(self) -> None:
        assert TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "Damaged item.\n\tBox crushed."})

    @pytest.mark.parametrize(
        "reason",
        [
            pytest.param("​" * 10, id="zero-width-spaces-only"),
            pytest.param("﻿" * 10, id="byte-order-marks-only"),
            pytest.param("dam​aged item", id="zero-width-space-inside"),
            pytest.param("abc‮def ghij", id="right-to-left-override"),
            pytest.param("abc⁦def⁩ ghij", id="bidi-isolate"),
            pytest.param("damaged‎ item", id="left-to-right-mark"),
            pytest.param("dam­aged item", id="soft-hyphen"),
            pytest.param("damaged⁠item ok", id="word-joiner"),
            pytest.param("Item damaged\U000e0049\U000e0047\U000e004e", id="tag-characters"),
        ],
    )
    def test_rejects_invisible_format_characters(self, reason: str) -> None:
        # Unicode category Cf: invisible, so they could satisfy the minimum length with nothing to
        # read, make the stored text display differently from what it contains, or hide text.
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason})
        assert error_types(exc) == {"format_characters"}

    @pytest.mark.parametrize(
        "reason",
        [
            pytest.param("\U0001f468‍\U0001f469‍\U0001f467 family order damaged", id="emoji-zwj-sequence"),
            pytest.param("می‌خواهم مرجوع", id="persian-zwnj"),
            pytest.param("क्‍ष damaged in transit", id="devanagari-zwj"),
        ],
    )
    def test_allows_joiners_between_visible_characters(self, reason: str) -> None:
        # ZWJ/ZWNJ are part of how Persian, Indic scripts and emoji are written.
        assert TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason}).reason == reason

    @pytest.mark.parametrize(
        "reason",
        [
            pytest.param("‍Item arrived damaged", id="leading-joiner"),
            pytest.param("Item arrived damaged‌", id="trailing-joiner"),
            pytest.param("Item arrived ‍ damaged", id="joiner-next-to-space"),
            pytest.param("ab‍‍‍‍‍‍‍‍cd", id="run-of-joiners"),
        ],
    )
    def test_rejects_joiners_that_join_nothing(self, reason: str) -> None:
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason})
        assert error_types(exc) == {"format_characters"}

    def test_is_nfc_normalized_before_the_length_check(self) -> None:
        # "é" spelled as "e" + U+0301 COMBINING ACUTE ACCENT is one character, not two.
        decomposed = "é" * 5 + "xxxxx"
        model = TriggerRefundInput.model_validate({**VALID_REFUND, "reason": decomposed})
        assert model.reason == "é" * 5 + "xxxxx"
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "é" * 9})
        assert error_types(exc) == {"string_too_short"}

    def test_length_is_counted_in_code_points(self) -> None:
        # An emoji outside the BMP is one code point (two UTF-16 units in a JavaScript client).
        assert TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "\U0001f4e6" * 10})
        with pytest.raises(ValidationError):
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "\U0001f4e6" * 501})

    @pytest.mark.parametrize("reason", [1234567890, None, ["Item arrived damaged"]])
    def test_rejects_non_strings(self, reason: Any) -> None:
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason})
        assert error_types(exc) == {"string_type"}


def test_reports_every_invalid_field_at_once() -> None:
    with pytest.raises(ValidationError) as exc:
        TriggerRefundInput.model_validate({"customer_id": "bad", "amount": -1, "reason": "short", "x": 1})
    assert {e["loc"][0] for e in exc.value.errors()} == {"customer_id", "amount", "reason", "x"}


def test_published_schema_matches_enforced_contract() -> None:
    schema = TriggerRefundInput.model_json_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"customer_id", "amount", "reason"}
    props = schema["properties"]
    assert props["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"
    assert props["customer_id"]["minLength"] == props["customer_id"]["maxLength"] == 10
    assert props["amount"]["type"] == "number"
    assert props["amount"]["exclusiveMinimum"] == 0
    assert props["amount"]["maximum"] == 1_000_000
    assert "multipleOf" not in props["amount"]  # float validators reject 0.07 as a multiple of 0.01
    assert props["reason"]["minLength"] == 10
    assert props["reason"]["maxLength"] == 500
    assert props["reason"]["pattern"] == REASON_PATTERN


def _schema_accepts(arguments: dict[str, Any]) -> bool:
    return Draft202012Validator(TriggerRefundInput.model_json_schema()).is_valid(arguments)


def _server_accepts(arguments: dict[str, Any]) -> bool:
    try:
        TriggerRefundInput.model_validate(arguments)
    except ValidationError:
        return False
    return True


# Each entry differs from VALID_REFUND in one field.
AGREED = [
    {},
    {"amount": 0.01},
    {"amount": 0.07},
    {"amount": 1_000_000},
    {"amount": 0},
    {"amount": -5},
    {"amount": 1_000_000.01},
    {"amount": "25.00"},
    {"amount": True},
    {"customer_id": "CUST-00000"},
    {"customer_id": "CUST-12345\n"},
    {"customer_id": "CUST-1234"},
    {"customer_id": "cust-12345"},
    {"customer_id": "CUST-" + "".join(chr(0x0661 + i) for i in range(5))},
    {"reason": "0123456789"},
    {"reason": "012345678"},
    {"reason": "x" * 500},
    {"reason": "x" * 501},
    {"reason": "Damaged item.\n\tBox crushed."},
    {"reason": "damaged item\x00 returned"},
    {"reason": "damaged\x1b[31m item returned"},
    {"reason": "line1\x85line2xx"},
    {"reason": "​" * 10},
    {"reason": "﻿" * 10},
    {"reason": "abc‮def ghij"},
    {"reason": "dam­aged item"},
    {"reason": "\U0001f468‍\U0001f469‍\U0001f467 family order damaged"},
    {"reason": "é" * 10},
    {"reason": "\U0001f4e6" * 10},
    {"reason": 1234567890},
    {"x": 1},
]


@pytest.mark.parametrize("change", AGREED, ids=lambda change: ascii(change)[:40])
def test_published_schema_and_server_agree(change: dict[str, Any]) -> None:
    arguments = {**VALID_REFUND, **change}
    assert _schema_accepts(arguments) == _server_accepts(arguments)


@pytest.mark.parametrize(
    "change",
    [
        {"reason": "  Item arrived damaged  "},
        {"reason": "Item arrived damaged\n"},
        {"reason": "\tItem arrived damaged"},
    ],
    ids=["padded", "trailing-newline", "leading-tab"],
)
def test_schema_is_stricter_than_the_server_only_about_edge_whitespace(change: dict[str, Any]) -> None:
    # The schema asks clients not to send surrounding whitespace; the server trims it instead.
    # (Python's `re` lets `$` match before a final newline, so `jsonschema` misses that one case.)
    arguments = {**VALID_REFUND, **change}
    assert _server_accepts(arguments)
    if not change["reason"].endswith("\n"):
        assert not _schema_accepts(arguments)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"amount": 0.001}, id="three-decimals"),
        pytest.param({"amount": 10.555}, id="three-decimals-2"),
        pytest.param({"amount": 0.1 + 0.2}, id="float-arithmetic"),
        pytest.param({"reason": "é" * 9}, id="nfc-shortens-below-minimum"),
        pytest.param({"reason": "Item damaged\U000e0049\U000e0047"}, id="astral-tag-characters"),
        pytest.param({"reason": "Item arrived damaged‍"}, id="joiner-joining-nothing"),
    ],
)
def test_server_only_rules(change: dict[str, Any]) -> None:
    # The rules JSON Schema cannot express (listed in the README): the schema accepts these, the
    # server does not. Every other difference runs the other way (edge whitespace, above).
    arguments = {**VALID_REFUND, **change}
    assert _schema_accepts(arguments)
    assert not _server_accepts(arguments)
