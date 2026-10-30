from __future__ import annotations

import pytest

from llm_router.keys import TenantKeyring, fingerprint_api_key, is_fingerprint
from llm_router.tokens import (
    MESSAGE_OVERHEAD_TOKENS,
    REPLY_PRIMING_TOKENS,
    accepted_usage,
    estimate_prompt_tokens,
    estimate_text_tokens,
    max_plausible_usage,
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


def test_plausible_usage_range() -> None:
    # max_tokens 500, prompt estimate 9: the provider cannot exceed 500 completion tokens, and the
    # prompt cannot be more than one token per byte, i.e. 4x its bytes/4 estimate.
    assert max_plausible_usage(9, 500) == 536
    for reported in (1, 9, 509, 536):
        assert accepted_usage(reported, prompt_tokens=9, max_tokens=500) == reported
    for reported in (None, 0, -1, 537, 10**15, 2**63):
        assert accepted_usage(reported, prompt_tokens=9, max_tokens=500) is None


def test_the_ceiling_covers_one_token_per_byte() -> None:
    # The worst case for a byte-level tokenizer: every byte its own token.
    for text in ("a a a a a a a a", "日本語のテキスト", "😀😀😀", "x" * 1001):
        prompt = estimate_prompt_tokens([{"role": "user", "content": text}])
        worst_case = len(text.encode()) + MESSAGE_OVERHEAD_TOKENS + REPLY_PRIMING_TOKENS
        assert accepted_usage(worst_case + 16, prompt_tokens=prompt, max_tokens=16) is not None


def test_fingerprints() -> None:
    fp = fingerprint_api_key("sk-test")
    assert is_fingerprint(fp)
    assert len(fp) == 64
    assert fp == fingerprint_api_key("sk-test")
    assert fp != fingerprint_api_key("sk-test2")
    assert not is_fingerprint("sk-test")
    assert not is_fingerprint(fp.upper())


def test_keyring_matches_bearer_tokens_by_fingerprint() -> None:
    keyring = TenantKeyring([fingerprint_api_key("key-a"), fingerprint_api_key("key-b").upper()])
    assert len(keyring) == 2
    assert keyring.authenticate("key-a") == fingerprint_api_key("key-a")
    assert keyring.authenticate_header("Bearer key-b") == fingerprint_api_key("key-b")
    assert keyring.authenticate_header("bearer   key-b ") == fingerprint_api_key("key-b")
    for header in (None, "", "Bearer", "Basic key-a", "Bearer key-c", "Bearer " + "k" * 600):
        assert keyring.authenticate_header(header) is None
    with pytest.raises(ValueError, match="fingerprint"):
        TenantKeyring(["key-a"])
