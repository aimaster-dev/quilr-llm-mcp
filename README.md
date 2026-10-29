# quilr-llm-mcp

Four small, runnable services that cover the security plumbing around AI agents:
**MCP servers**, an **MCP security gateway**, and two pieces of an **LLM gateway** (a streaming
PII guardrail and a rate-limiting/fallback router).

Each task is an independent Python 3.12 project with its own `pyproject.toml`, README, and test suite,
so you can review, install, and run them one at a time.

| # | Project | What it shows | Status |
|---|---------|---------------|--------|
| 1 | [`task1-mcp-server/`](task1-mcp-server/README.md) | MCP server on the official SDK; strict Pydantic validation; stdout reserved for JSON-RPC | done |
| 2 | [`task2-mcp-gateway/`](task2-mcp-gateway/README.md) | HTTP/JSON-RPC reverse proxy with Bearer JWT auth and `admin_*` tool authorization | done |
| 3 | [`task3-llm-stream-guardrail/`](task3-llm-stream-guardrail/README.md) | OpenAI-compatible streaming proxy that redacts PII across chunk boundaries | done |
| 4 | [`task4-llm-router/`](task4-llm-router/README.md) | Token-aware sliding-window limiter on SQLite, 3000 ms timeout, 429 failover | done |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # installs every task project in editable mode
```

Run a single task's tests, or all of them:

```bash
cd task1-mcp-server && pytest
for d in task*/; do (cd "$d" && pytest -q); done
```

## Try it

Each task README has a full walkthrough with real output. The services start with:

| Task | Commands |
|------|----------|
| 1 | `customer-mcp` speaks JSON-RPC on stdin/stdout; point an MCP host at it (config in the task README) |
| 2 | `mock-mcp-server --port 9000`, then `mcp-gateway --downstream-url http://127.0.0.1:9000/mcp --port 8000` (needs `MCP_GATEWAY_JWT_SECRET`); mint tokens with `mcp-gateway-token --subject alice --role viewer` |
| 3 | `stream-guardrail-mock --port 9000`, then `stream-guardrail --port 8080`; stream with `curl -N` |
| 4 | `mock-llm-provider --name primary --port 9001` and `--name secondary --port 9002`, then `llm-router` with the provider URLs (needs `LLM_ROUTER_TENANT_KEYS`) |

## Repository conventions

- **Issues and milestones**: every task has a milestone, and each unit of work is an issue under it.
- **Branches**: `feature/task-<n>-<topic>` off `main`, merged with a pull request.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) with a task scope,
  e.g. `feat(task2): reject admin_* tool calls for non-admin roles`, each referencing its issue.
- **CI**: GitHub Actions runs `ruff` and each task's `pytest` suite on Ubuntu and Windows.
  New `task*/` projects are picked up automatically. A final `CI passed` job fails if any other
  job failed, and it is the single status check that branch protection on `main` requires.
