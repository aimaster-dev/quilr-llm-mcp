"""A mock OpenAI-compatible provider for demos and end-to-end tests (``mock-llm-provider``).

Behaviour is selected with ``--mode`` and can be switched at runtime with
``PUT /_mock/mode {"mode": "...", "delay": 5}``. ``GET /_mock/stats`` reports request and
client-disconnect counts.

========  =================================================================================
``ok``    200 with a completion; ``usage`` reports the full ``max_tokens`` as generated
``429``   429 with ``Retry-After`` and a body full of internal details
``500``   500 with a fake Python traceback, an internal hostname and a fake key
``400``   400 ``invalid_request_error`` (a non-retryable rejection)
``slow``  waits ``delay`` seconds, then answers 200 (stops early if the client hangs up)
``drip``  sends headers at once, then the 200 body one byte at a time over ``delay`` seconds
========  =================================================================================

The error bodies deliberately contain things a gateway must never relay to its clients.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_router.tokens import estimate_prompt_tokens, estimate_text_tokens

MODES = ("ok", "429", "500", "400", "slow", "drip")
INTERNAL_HOST = "gpu-node-7.inference.mock.internal"
FAKE_SECRET = "sk-mock-internal-0000-DO-NOT-LEAK"  # noqa: S105 - deliberately fake, for leak tests


@dataclass
class MockState:
    name: str
    mode: str
    delay: float
    requests: int = 0
    disconnects: int = 0


def _completion(state: MockState, body: dict[str, Any], number: int) -> dict[str, Any]:
    messages = body.get("messages")
    messages = [m for m in messages if isinstance(m, dict)] if isinstance(messages, list) else []
    max_tokens = body.get("max_tokens") if isinstance(body.get("max_tokens"), int) else None
    content = f"Hello from the {state.name} mock provider (request #{number})."
    prompt_tokens = estimate_prompt_tokens(messages)
    # Report the whole completion budget as used (a worst case), so demos reach the limit quickly.
    completion_tokens = max_tokens or estimate_text_tokens(content)
    return {
        "id": f"chatcmpl-mock-{state.name}-{number}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock-model"),
        "system_fingerprint": f"fp_{state.name}_{INTERNAL_HOST}",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _sleep_unless_disconnected(request: Request, seconds: float) -> bool:
    """Sleep, polling for a client disconnect. Returns False if the client went away."""
    deadline = anyio.current_time() + seconds
    while (left := deadline - anyio.current_time()) > 0:
        if await request.is_disconnected():
            return False
        await anyio.sleep(min(0.02, left))
    return True


def create_mock_app(*, name: str = "mock", mode: str = "ok", delay: float = 5.0) -> Starlette:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    state = MockState(name=name, mode=mode, delay=delay)

    async def completions(request: Request) -> Response:
        state.requests += 1
        number = state.requests
        try:
            body = json.loads(await request.body() or b"{}")
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}

        match state.mode:
            case "429":
                return JSONResponse(
                    {
                        "error": {
                            "message": f"Rate limit reached for {state.name} on shard {INTERNAL_HOST}",
                            "type": "requests",
                            "code": "rate_limit_exceeded",
                        }
                    },
                    status_code=429,
                    headers={"Retry-After": "1"},
                )
            case "500":
                traceback = (
                    "Traceback (most recent call last):\n"
                    f'  File "/srv/{state.name}/inference/server.py", line 812, in generate\n'
                    f"    client = connect('{INTERNAL_HOST}', api_key='{FAKE_SECRET}')\n"
                    "RuntimeError: CUDA error: out of memory\n"
                )
                return PlainTextResponse(traceback, status_code=500)
            case "400":
                return JSONResponse(
                    {
                        "error": {
                            "message": f"Invalid 'messages' for {state.name} ({INTERNAL_HOST}).",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=400,
                )
            case "slow":
                if not await _sleep_unless_disconnected(request, state.delay):
                    state.disconnects += 1
                    return Response(status_code=204)  # nobody is listening any more
            case "drip":
                return StreamingResponse(
                    _drip(_completion(state, body, number)), media_type="application/json"
                )
        return JSONResponse(_completion(state, body, number))

    async def _drip(completion: dict[str, Any]) -> AsyncIterator[bytes]:
        data = json.dumps(completion).encode()
        # Small pieces every >= 50 ms: shorter sleeps are unreliable on coarse (Windows) timers.
        pieces = max(1, min(len(data), int(state.delay / 0.05)))
        size = -(-len(data) // pieces)  # ceiling division
        try:
            for start in range(0, len(data), size):
                yield data[start : start + size]
                await anyio.sleep(state.delay / pieces)
        except BaseException:
            state.disconnects += 1  # cancelled because the client hung up mid-body
            raise

    async def set_mode(request: Request) -> Response:
        try:
            update = await request.json()
            new_mode = update.get("mode", state.mode)
            new_delay = float(update.get("delay", state.delay))
        except (ValueError, TypeError, AttributeError):
            return JSONResponse({"error": 'expected {"mode": ..., "delay": ...}'}, status_code=400)
        if new_mode not in MODES or new_delay <= 0:
            return JSONResponse({"error": f"mode must be one of {list(MODES)}, delay > 0"}, status_code=400)
        state.mode, state.delay = new_mode, new_delay
        return JSONResponse(asdict(state))

    async def stats(request: Request) -> Response:
        return JSONResponse(asdict(state))

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", completions, methods=["POST"]),
            Route("/_mock/mode", set_mode, methods=["PUT"]),
            Route("/_mock/stats", stats, methods=["GET"]),
        ]
    )
    app.state.mock = state
    return app


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mock-llm-provider", description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="mock", help="label used in responses (default: mock)")
    parser.add_argument("--mode", choices=MODES, default="ok", help="initial behaviour (default: ok)")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds for slow/drip modes (default: 5)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args(argv)
    app = create_mock_app(name=args.name, mode=args.mode, delay=args.delay)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
