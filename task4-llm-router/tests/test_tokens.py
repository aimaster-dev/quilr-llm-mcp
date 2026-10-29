from __future__ import annotations

import pytest

from llm_router.keys import fingerprint_api_key, is_fingerprint
from llm_router.tokens import (
    MESSAGE_OVERHEAD_TOKENS,
    REPLY_PRIMING_TOKENS,
    estimate_prompt_tokens,
    estimate_text_tokens,
    request_cost,
)


@pytest.mark.parametrize(
    ("text", "tokens"),
    [("", 0), ("abcd", 1), ("abcde", 2), ("x" * 400, 100), ("日本語", 3), ("😀", 1)],
)
def test_text_estimate_is_utf8_bytes_over_four_rounded_up(text: str, tokens: int) -> None:
    assert estimate_text_tokens(text) == tokens


def test_prompt_estimate_counts_message_overhead() -> None:
    messages = [
        {"role": "system", "content": "x" * 40},
        {"role": "user", "content": [{"type": "text", "text": "y" * 20}, {"type": "text", "text": "z" * 20}]},
        {"role": "assistant", "content": None},
    ]
    expected = REPLY_PRIMING_TOKENS + 3 * MESSAGE_OVERHEAD_TOKENS + 10 + 10
    assert estimate_prompt_tokens(messages) == expected


def test_named_messages_include_the_name() -> None:
    plain = estimate_prompt_tokens([{"role": "user", "content": "hi"}])
    named = estimate_prompt_tokens([{"role": "user", "content": "hi", "name": "alice-the-user"}])
    assert named == plain + estimate_text_tokens("alice-the-user")


def test_request_cost_adds_the_completion_budget() -> None:
    assert request_cost(120, 256) == 376
    with pytest.raises(ValueError):
        request_cost(-1, 10)


def test_fingerprints() -> None:
    fp = fingerprint_api_key("sk-test")
    assert is_fingerprint(fp)
    assert len(fp) == 64
    assert fp == fingerprint_api_key("sk-test")
    assert fp != fingerprint_api_key("sk-test2")
    assert not is_fingerprint("sk-test")
    assert not is_fingerprint(fp.upper())
