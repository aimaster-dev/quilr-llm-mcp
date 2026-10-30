"""Tool behaviour through the real MCP protocol layer (official SDK client, in-process)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import mcp_types as types
import pytest
from jsonschema import Draft202012Validator
from mcp import Client
from mcp.shared.exceptions import MCPError

from customer_mcp.schemas import ToolErrorText
from customer_mcp.server import create_server
from customer_mcp.store import CustomerStore

pytestmark = pytest.mark.anyio

TOOL_ERROR_SCHEMA = ToolErrorText.model_json_schema()


@pytest.fixture
async def client() -> AsyncIterator[Client]:
    async with Client(create_server(CustomerStore.with_seed_data())) as connected:
        yield connected


async def call_expecting_error(client: Client, name: str, arguments: dict[str, Any]) -> MCPError:
    with pytest.raises(MCPError) as exc:
        await client.call_tool(name, arguments)
    return exc.value


def tool_error(result: types.CallToolResult) -> dict[str, str]:
    """The error of an `isError` result: JSON text content, and no structured content."""
    assert result.is_error is True
    assert result.structured_content is None
    assert isinstance(result.content[0], types.TextContent)
    payload = json.loads(result.content[0].text)
    Draft202012Validator(TOOL_ERROR_SCHEMA).validate(payload)
    return payload["error"]


async def test_lists_both_tools_with_strict_schemas(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == {"get_customer_record", "trigger_refund"}

    refund = tools["trigger_refund"]
    assert refund.input_schema["additionalProperties"] is False
    assert refund.input_schema["required"] == ["customer_id", "amount", "reason"]
    assert refund.output_schema is not None
    assert refund.annotations is not None
    assert refund.annotations.destructive_hint is True
    lookup = tools["get_customer_record"]
    assert lookup.annotations is not None
    assert lookup.annotations.read_only_hint is True
    # Both work on the same in-memory store, so they describe the same (closed) world.
    assert lookup.annotations.open_world_hint is refund.annotations.open_world_hint is False


async def test_get_customer_record(client: Client) -> None:
    result = await client.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
    assert result.is_error is False
    record = result.structured_content
    assert record["name"] == "Ada Lovelace"
    assert record["refundable_balance"] == "1250.00"
    assert isinstance(result.content[0], types.TextContent)


async def test_unknown_customer_is_a_tool_execution_error(client: Client) -> None:
    result = await client.call_tool("get_customer_record", {"customer_id": "CUST-99999"})
    assert tool_error(result) == {
        "code": "customer_not_found",
        "message": "Customer CUST-99999 was not found",
    }


async def test_every_result_conforms_to_the_advertised_schemas(client: Client) -> None:
    # MCP: structured results MUST conform to the tool's outputSchema. Successful results carry
    # structuredContent that validates against it; error results carry none (a client such as the
    # TypeScript SDK validates any structuredContent it gets), and their text is the error schema.
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    calls: list[tuple[str, dict[str, Any], bool]] = [
        ("get_customer_record", {"customer_id": "CUST-10001"}, False),
        ("get_customer_record", {"customer_id": "CUST-99999"}, True),
        (
            "trigger_refund",
            {"customer_id": "CUST-10002", "amount": 10.5, "reason": "Duplicate charge"},
            False,
        ),
        ("trigger_refund", {"customer_id": "CUST-10004", "amount": 1, "reason": "Duplicate charge"}, True),
        ("trigger_refund", {"customer_id": "CUST-10002", "amount": 999, "reason": "Duplicate charge"}, True),
    ]
    for name, arguments, fails in calls:
        output_schema = tools[name].output_schema
        assert output_schema is not None
        Draft202012Validator.check_schema(output_schema)
        result = await client.call_tool(name, arguments)
        if fails:
            tool_error(result)
        else:
            assert result.is_error is False
            Draft202012Validator(output_schema).validate(result.structured_content)
            assert isinstance(result.content[0], types.TextContent)
            assert json.loads(result.content[0].text) == result.structured_content


async def test_refund_reduces_refundable_balance(client: Client) -> None:
    result = await client.call_tool(
        "trigger_refund",
        {"customer_id": "CUST-10002", "amount": 10.5, "reason": "  Duplicate charge on invoice  "},
    )
    assert result.is_error is False
    receipt = result.structured_content
    assert receipt["amount"] == "10.50"
    assert receipt["reason"] == "Duplicate charge on invoice"
    assert receipt["remaining_refundable_balance"] == "300.00"
    assert receipt["refund_id"].startswith("RF-")

    record = await client.call_tool("get_customer_record", {"customer_id": "CUST-10002"})
    assert record.structured_content["refundable_balance"] == "300.00"


@pytest.mark.parametrize(
    ("customer_id", "amount", "code"),
    [
        ("CUST-10002", 310.51, "exceeds_refundable_balance"),
        ("CUST-10004", 5.0, "customer_not_active"),
        ("CUST-10005", 5.0, "customer_not_active"),
        ("CUST-99999", 5.0, "customer_not_found"),
    ],
)
async def test_refund_business_rules(client: Client, customer_id: str, amount: float, code: str) -> None:
    result = await client.call_tool(
        "trigger_refund",
        {"customer_id": customer_id, "amount": amount, "reason": "Customer requested refund"},
    )
    assert tool_error(result)["code"] == code


async def test_invalid_arguments_are_json_rpc_invalid_params(client: Client) -> None:
    error = await call_expecting_error(
        client,
        "trigger_refund",
        {"customer_id": "CUST-1", "amount": "25", "reason": "short", "admin": True},
    )
    assert error.code == types.INVALID_PARAMS
    assert error.message == "Invalid arguments for tool 'trigger_refund'"
    fields = {item["field"]: item["type"] for item in error.data["errors"]}
    assert fields == {
        "customer_id": "string_pattern_mismatch",
        "amount": "float_type",
        "reason": "string_too_short",
        "admin": "extra_forbidden",
    }


async def test_error_details_never_echo_input_values(client: Client) -> None:
    secret = "4111111111111111-ssn-123-45-6789"
    error = await call_expecting_error(client, "get_customer_record", {"customer_id": secret})
    assert error.code == types.INVALID_PARAMS
    assert secret not in repr(error.data)
    assert secret not in error.message


async def test_missing_arguments_are_invalid_params(client: Client) -> None:
    error = await call_expecting_error(client, "trigger_refund", {})
    assert error.code == types.INVALID_PARAMS
    assert {item["field"] for item in error.data["errors"]} == {"customer_id", "amount", "reason"}


async def test_unknown_tool_is_invalid_params(client: Client) -> None:
    error = await call_expecting_error(client, "delete_customer", {"customer_id": "CUST-10001"})
    assert error.code == types.INVALID_PARAMS
    assert error.data == {"tool": "delete_customer"}


async def test_concurrent_refunds_never_overdraw_balance() -> None:
    # CUST-10002 has 310.50 refundable: 20 concurrent refunds of 20.00 -> exactly 15 succeed.
    async with Client(create_server(CustomerStore.with_seed_data())) as client:
        outcomes: list[bool] = []

        async def refund() -> None:
            result = await client.call_tool(
                "trigger_refund",
                {"customer_id": "CUST-10002", "amount": 20, "reason": "Concurrent refund test"},
            )
            outcomes.append(not result.is_error)

        async with anyio.create_task_group() as tg:
            for _ in range(20):
                tg.start_soon(refund)

        assert outcomes.count(True) == 15
        record = await client.call_tool("get_customer_record", {"customer_id": "CUST-10002"})
        assert record.structured_content["refundable_balance"] == "10.50"
