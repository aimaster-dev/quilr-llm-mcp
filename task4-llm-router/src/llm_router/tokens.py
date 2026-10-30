"""Token cost estimation without a tokenizer dependency.

The limiter needs a cost *before* the request is sent, so it reserves an estimate and corrects it
later with the provider's reported `usage`. The heuristic is deliberately simple:

* text: ``ceil(utf8_bytes / 4)``. Four bytes per token is the usual rule of thumb for English with
  BPE tokenizers. Counting UTF-8 bytes instead of characters keeps CJK and emoji text (3-4 bytes per
  character, often one or more tokens each) from being under-counted by up to 4x.
* each chat message adds a fixed overhead for the role and separators, and the reply is primed
  with a few tokens, mirroring the OpenAI chat format.

A request's cost is ``prompt_estimate + max_tokens``: the provider can never generate more than
``max_tokens``, so the reservation is an upper bound unless the prompt estimate is too low.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

BYTES_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
REPLY_PRIMING_TOKENS = 3


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text.encode("utf-8")) / BYTES_PER_TOKEN)


def _content_text(content: Any) -> str:
    """Flatten OpenAI-style message content (a string or a list of text parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif isinstance(part, str):
            parts.append(part)
    return "".join(parts)


def estimate_prompt_tokens(messages: Iterable[Mapping[str, Any]]) -> int:
    """Estimate the prompt tokens of a list of chat messages."""
    total = REPLY_PRIMING_TOKENS
    for message in messages:
        total += MESSAGE_OVERHEAD_TOKENS
        total += estimate_text_tokens(_content_text(message.get("content")))
        name = message.get("name")
        if isinstance(name, str):
            total += estimate_text_tokens(name)
    return total


def request_cost(prompt_tokens: int, max_tokens: int) -> int:
    """Tokens to reserve for a request: the prompt estimate plus the completion budget."""
    if prompt_tokens < 0 or max_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return prompt_tokens + max_tokens


def max_plausible_usage(prompt_tokens: int, max_tokens: int) -> int:
    """The most tokens a correct provider can report for a request.

    * The completion is capped by ``max_tokens``, which the gateway always sends upstream.
    * A token covers at least one byte of text (byte-level BPE and byte-fallback tokenizers), so
      the real prompt is at most ``BYTES_PER_TOKEN`` times its ``bytes / BYTES_PER_TOKEN``
      estimate. Multiplying the per-message overhead too leaves room for chat-template tokens.
    """
    return max_tokens + BYTES_PER_TOKEN * prompt_tokens


def accepted_usage(reported: int | None, *, prompt_tokens: int, max_tokens: int) -> int | None:
    """The provider-reported total to charge, or ``None`` to keep the gateway's own reservation.

    A report is accepted only if a correct provider could have produced it for this request:
    ``1 <= reported <= max_plausible_usage(...)``. Anything else (zero, which some servers send
    when they do not count, negative, absurdly large, missing) says nothing about the real usage,
    so neither it nor a clamped version of it is charged. The reservation (prompt estimate plus
    ``max_tokens``) is the gateway's own bound and is kept instead: a broken provider can neither
    make requests free nor lock a tenant out.
    """
    if reported is None or not 1 <= reported <= max_plausible_usage(prompt_tokens, max_tokens):
        return None
    return reported
