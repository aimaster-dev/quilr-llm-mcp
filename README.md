# quilr-llm-mcp

Four small, runnable services that cover the security plumbing around AI agents:
**MCP servers**, an **MCP security gateway**, and two pieces of an **LLM gateway** (a streaming
PII guardrail and a rate-limiting/fallback router).

Each task is an independent Python 3.12 project with its own `pyproject.toml`, README, and test suite,
so you can review, install, and run them one at a time.

| # | Project | What it shows | Status |
|---|---------|---------------|--------|
| 1 | `task1-mcp-server/` | MCP server on the official SDK; strict Pydantic validation; stdout reserved for JSON-RPC | planned |
| 2 | `task2-mcp-gateway/` | HTTP/JSON-RPC reverse proxy with Bearer auth and `admin_*` tool authorization | planned |
| 3 | `task3-llm-stream-guardrail/` | OpenAI-compatible streaming proxy that redacts PII across chunk boundaries | planned |
| 4 | `task4-llm-router/` | Token-aware sliding-window limiter on SQLite, 3000 ms timeout, 429 failover | planned |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # installs every task project in editable mode
```

Then follow the README inside each task folder. Run a single task's tests with:

```bash
cd task1-mcp-server && pytest
```

## Repository conventions

- **Issues and milestones**: every task has a milestone, and each unit of work is an issue under it.
- **Branches**: `feature/task-<n>-<topic>` off `main`, merged with a pull request.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) with a task scope,
  e.g. `feat(task2): reject admin_* tool calls for non-admin roles`, each referencing its issue.
- **CI**: GitHub Actions runs `ruff` and each task's `pytest` suite on Ubuntu and Windows.
  New `task*/` projects are picked up automatically. A final `CI passed` job fails if any other
  job failed, and it is the single status check that branch protection on `main` requires.
