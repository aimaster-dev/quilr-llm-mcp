"""Edge cases for the strict input contracts."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from customer_mcp.schemas import GetCustomerRecordInput, TriggerRefundInput

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
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"customer_id", "amount", "reason"}
    props = schema["properties"]
    assert props["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"
    assert props["amount"]["type"] == "number"
    assert props["amount"]["exclusiveMinimum"] == 0
    assert props["reason"]["minLength"] == 10
