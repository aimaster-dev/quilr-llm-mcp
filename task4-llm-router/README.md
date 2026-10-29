# Task 4 — LLM gateway router (rate limiting + model failover)

A model-routing module for an LLM gateway: a per-tenant, token-aware sliding-window rate limiter
whose state lives in an **on-disk SQLite database**.

## Rate limiter

Each tenant API key may use **50,000 tokens per rolling 60 s window**. A request reserves
`prompt_estimate + max_tokens` tokens before it is sent upstream; the reservation is corrected with
the provider's reported `usage` afterwards, or released if no completion was produced.

```bash
pip install -e ".[dev]"
pytest
```
