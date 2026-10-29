"""Benchmarks for the streaming PII guardrail.

    python benchmarks/bench_redactor.py            # about 30 s
    python benchmarks/bench_redactor.py --quick    # fewer iterations, for a smoke run

1. Engine: cost of ``StreamRedactor.feed()`` per chunk, by chunk size, against a pass-through.
2. Gateway pipeline: CPU per upstream event for SSE parse -> JSON -> redact -> JSON -> SSE,
   against the same proxy pipeline without redaction (parse -> encode).
3. End to end over loopback TCP, with the mock provider and the gateway on uvicorn in this process:
   time to first byte, time to first content and total time, measured directly against the mock,
   through the gateway with redaction disabled, and through the gateway with redaction.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable
from multiprocessing.queues import Queue
from typing import Any

import anyio
import httpx

from stream_guardrail.completions import ChatStreamRedactor
from stream_guardrail.config import Settings
from stream_guardrail.gateway import create_app
from stream_guardrail.mock_provider import DEFAULT_TEXT, MockProvider, MockSettings
from stream_guardrail.redaction import StreamRedactor
from stream_guardrail.sse import SSEEvent, SSEParser
from stream_guardrail.testing import serve

PROSE = (
    "The deployment finished at 14:05 and all 12 health checks passed. Latency stayed under 40 ms "
    "for 99% of requests, and the error budget for the quarter is still at 87%. Next steps: rotate "
    "the staging credentials, archive version 2.3.1 and update the runbook. "
)


def corpus(size: int) -> str:
    """LLM-like text: mostly prose, with a PII-laden paragraph every ~1 KB."""
    rng = random.Random(7)
    parts, total = [], 0
    while total < size:
        part = DEFAULT_TEXT if rng.random() < 0.2 else PROSE
        parts.append(part)
        total += len(part)
    return "".join(parts)[:size]


def identity(chunk: str) -> str:
    return chunk


def best_of(runs: int, fn: Callable[[], None]) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def bench_engine(text: str, runs: int) -> None:
    print(f"\n## 1. Engine: StreamRedactor.feed() on {len(text) / 1e3:.0f} KB of LLM-like text\n")
    print("| chunk size | chunks | pass-through µs/chunk | redactor µs/chunk | overhead µs/chunk | MB/s |")
    print("|---:|---:|---:|---:|---:|---:|")
    for size in (1, 4, 16, 64, 256, 4096):
        chunks = [text[i : i + size] for i in range(0, len(text), size)]

        def passthrough(chunks: list[str] = chunks) -> None:
            out = [identity(c) for c in chunks]  # one Python call per chunk, like any proxy
            "".join(out)

        def redacting(chunks: list[str] = chunks) -> None:
            redactor = StreamRedactor()
            out = [redactor.feed(c) for c in chunks]
            out.append(redactor.flush())
            "".join(out)

        base = best_of(runs, passthrough)
        red = best_of(runs, redacting)
        n = len(chunks)
        print(
            f"| {size} | {n:,} | {base / n * 1e6:.2f} | {red / n * 1e6:.2f} | "
            f"{(red - base) / n * 1e6:.2f} | {len(text) / red / 1e6:.1f} |"
        )


class PassThrough(ChatStreamRedactor):
    """Stands in for the real redactor to measure the same proxy without redaction."""

    def process(self, event: SSEEvent) -> list[SSEEvent]:
        return [event]

    def finish(self) -> list[SSEEvent]:
        return []


def sse_stream(text: str, chunk_size: int) -> list[bytes]:
    events = []
    for i in range(0, len(text), chunk_size):
        payload = {
            "id": "chatcmpl-bench",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "mock-llm-1",
            "choices": [{"index": 0, "delta": {"content": text[i : i + chunk_size]}, "finish_reason": None}],
        }
        events.append(b"data: " + json.dumps(payload).encode() + b"\n\n")
    return events


def bench_pipeline(text: str, runs: int) -> None:
    events = sse_stream(text, 4)  # ~1 token per event, like a real provider
    print(f"\n## 2. Gateway pipeline per upstream event ({len(events):,} events of 4 chars)\n")

    def run(make: Callable[[], Any]) -> None:
        parser, redactor = SSEParser(), make()
        for raw in events:
            b"".join(e.encode() for ev in parser.feed(raw) for e in redactor.process(ev))
        redactor.finish()

    base = best_of(runs, lambda: run(PassThrough))
    red = best_of(runs, lambda: run(ChatStreamRedactor))
    n = len(events)
    print("| pipeline | µs/event |")
    print("|---|---:|")
    print(f"| parse + re-encode (pass-through proxy) | {base / n * 1e6:.1f} |")
    print(f"| parse + JSON + redact + re-encode (gateway) | {red / n * 1e6:.1f} |")
    print(f"| **added by the guardrail** | **{(red - base) / n * 1e6:.1f}** |")


async def timed_request(
    client: httpx.AsyncClient, url: str, body: dict[str, Any]
) -> tuple[float, float, float]:
    """Return (time to first byte, time to first content, total) in milliseconds."""
    parser = SSEParser()
    start = time.perf_counter()
    first_byte = first_content = None
    async with client.stream("POST", url, json=body) as response:
        async for raw in response.aiter_bytes():
            now = time.perf_counter()
            if first_byte is None:
                first_byte = now
            if first_content is None:
                for event in parser.feed(raw):
                    if event.data and event.data != "[DONE]":
                        for choice in json.loads(event.data).get("choices", []):
                            if choice.get("delta", {}).get("content"):
                                first_content = now
    end = time.perf_counter()
    assert first_byte is not None and first_content is not None
    return (first_byte - start) * 1e3, (first_content - start) * 1e3, (end - start) * 1e3


def summary(samples: list[float]) -> str:
    samples = sorted(samples)
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
    return f"{statistics.median(samples):.2f} / {p95:.2f}"


def run_servers(mock_settings: MockSettings, urls: Queue, stop: Any) -> None:
    """Child process: the mock plus two gateways (redaction on / off) in front of it."""

    async def main() -> None:
        provider = MockProvider(mock_settings)
        async with serve(provider.app) as mock_url:
            upstream = Settings(upstream_base_url=f"{mock_url}/v1")
            async with (
                serve(create_app(upstream)) as on_url,
                serve(create_app(upstream, redactor_factory=PassThrough)) as off_url,
            ):
                urls.put((mock_url, off_url, on_url))
                await anyio.to_thread.run_sync(stop.wait)  # a multiprocessing.Event cannot be awaited

    anyio.run(main)


async def measure(urls: tuple[str, str, str], n: int) -> dict[str, list[tuple[float, float, float]]]:
    mock_url, off_url, on_url = urls
    targets = {
        "direct to mock": mock_url,
        "gateway, redaction off": off_url,
        "gateway, redaction on": on_url,
    }
    body = {"model": "mock-llm-1", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    results: dict[str, list[tuple[float, float, float]]] = {name: [] for name in targets}
    async with httpx.AsyncClient(timeout=30) as client:
        for url in targets.values():  # warm up connections
            await timed_request(client, f"{url}/v1/chat/completions", body)
        for _ in range(n):
            for name, url in targets.items():  # interleaved, so drift hits every path equally
                results[name].append(await timed_request(client, f"{url}/v1/chat/completions", body))
    return results


def bench_end_to_end(requests: int, paced_requests: int) -> None:
    scenarios = [
        (
            "unpaced (mock answers instantly)",
            MockSettings(chunk_size=4, delay=0, first_token_delay=0),
            requests,
        ),
        (
            "paced (TTFT 50 ms, then a 4-char delta every 20 ms)",
            MockSettings(text=DEFAULT_TEXT[:48], chunk_size=4, delay=0.02, first_token_delay=0.05),
            paced_requests,
        ),
    ]
    # Servers run in a child process so the client's timing is not skewed by the servers' work.
    ctx = multiprocessing.get_context("spawn")
    for title, mock_settings, n in scenarios:
        urls: Queue = ctx.Queue()
        stop = ctx.Event()
        child = ctx.Process(target=run_servers, args=(mock_settings, urls, stop), daemon=True)
        child.start()
        try:
            results = anyio.run(measure, urls.get(timeout=60), n)
        finally:
            stop.set()
            child.join(10)
        print(f"\n## 3. End to end over loopback TCP: {title}, {n} requests each\n")
        print("| path | first byte ms (p50 / p95) | first content ms (p50 / p95) | total ms (p50 / p95) |")
        print("|---|---:|---:|---:|")
        for name, rows in results.items():
            ttfb, ttfc, total = zip(*rows, strict=True)
            print(f"| {name} | {summary(list(ttfb))} | {summary(list(ttfc))} | {summary(list(total))} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="fewer iterations")
    args = parser.parse_args()
    runs, requests, paced = (1, 20, 3) if args.quick else (5, 200, 20)
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    text = corpus(200_000)
    bench_engine(text, runs)
    bench_pipeline(text[:50_000], runs)
    bench_end_to_end(requests, paced)


if __name__ == "__main__":
    main()
