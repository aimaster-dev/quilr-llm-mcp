"""Streaming PII guardrail for LLM responses."""

from stream_guardrail.redaction import MAX_BUFFERED, REDACTED, StreamRedactor, redact

__version__ = "0.1.0"

__all__ = ["MAX_BUFFERED", "REDACTED", "StreamRedactor", "__version__", "redact"]
