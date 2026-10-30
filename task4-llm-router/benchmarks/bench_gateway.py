"""Benchmarks behind the numbers in the README.

    python benchmarks/bench_gateway.py limiter                  # SQLite limiter alone, ~5 s
    python benchmarks/bench_gateway.py throughput -c 10 -n 2000
    python benchmarks/bench_gateway.py throughput -c 10 -n 2000 --direct
    python benchmarks/bench_gateway.py failover -c 1 100 200 400

1. ``limiter``: reserve + reconcile pairs per second on one connection (the hot path of a served
   request), then the async facade with many concurrent callers queueing for its thread.
2. ``throughput``: end to end over loopback TCP. A healthy mock primary and ``llm-router`` (one
   worker) run as separate processes; this process is the load generator (httpx, ``-c``
   concurrent requests, ``-n`` in total). ``--direct`` sends the same load to the mock provider
   itself: the ceiling of the rig (Python client, uvicorn server, same machine), to compare with.
3. ``failover``: the primary never answers in time (mock ``slow`` mode). For each concurrency level
   it reports the client-side latency and, from the gateway's own log, how long each primary
   attempt took before the 3000 ms deadline cancelled it, i.e. the deadline overshoot.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import anyio
import httpx

from llm_router.keys import fingerprint_api_key
from llm_router.limiter import AsyncTokenRateLimiter, TokenRateLimiter

TENANT_KEY = "bench-tenant-key"
BODY = {"model": "gpt-bench", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


# -- 1. the limiter alone ------------------------------------------------------------------------


def bench_limiter(pairs: int, concurrency: int) -> None:
    """Every reservation sums the tenant's live rows, so the cost grows with them. Two shapes:
    many tenants with a few live rows each, and one tenant whose window fills with ``pairs`` rows
    (at 50,000 tokens per window and ~10-token requests that is the worst case, ~5,000 rows)."""
    tenants = [fingerprint_api_key(f"{TENANT_KEY}-{i}") for i in range(100)]
    for label, rotation in (("100 tenants", tenants), ("1 tenant ", tenants[:1])):
        with tempfile.TemporaryDirectory() as tmp:
            limiter = TokenRateLimiter(Path(tmp) / "bench.db", limit_tokens=2**40)
            try:
                started = time.perf_counter()
                for i in range(pairs):
                    tenant = rotation[i % len(rotation)]
                    limiter.reconcile(limiter.reserve(tenant, 10).reservation_id, 5)
                elapsed = time.perf_counter() - started
                live = pairs // len(rotation)
                print(
                    f"sync reserve+reconcile, {label} (up to {live:,} live rows each): "
                    f"{pairs / elapsed:,.0f} pairs/s ({elapsed / pairs * 1e3:.3f} ms per pair)"
                )
                if len(rotation) > 1:
                    anyio.run(bench_async_facade, limiter, rotation, concurrency)
            finally:
                limiter.close()


async def bench_async_facade(limiter: TokenRateLimiter, tenants: list[str], concurrency: int) -> None:
    facade = AsyncTokenRateLimiter(limiter)
    latencies: list[float] = []

    async def one(tenant: str) -> None:
        t0 = time.perf_counter()
        decision = await facade.reserve(tenant, 10)
        latencies.append(time.perf_counter() - t0)
        await facade.reconcile(decision.reservation_id, 5)

    t0 = time.perf_counter()
    async with anyio.create_task_group() as tg:
        for i in range(concurrency):
            tg.start_soon(one, tenants[i % len(tenants)])
    total = time.perf_counter() - t0
    print(
        f"async facade, {concurrency} concurrent callers: {concurrency / total:,.0f} pairs/s; "
        f"reserve latency p50={percentile(latencies, 0.5) * 1e3:.1f} ms "
        f"max={max(latencies) * 1e3:.1f} ms (queueing for the single limiter thread)"
    )


# -- running the gateway and the mock providers as real processes --------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_up(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=1, trust_env=False)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"{url} did not come up")


@contextmanager
def stack(tmp: Path, primary_mode: str, log_level: str) -> Iterator[tuple[str, str, Path]]:
    """Mock primary + mock secondary + llm-router (one worker), each in its own process.

    Yields the gateway's URL, the primary mock's URL and the gateway's log file.
    """
    primary, secondary, gateway = free_port(), free_port(), free_port()
    processes: list[subprocess.Popen] = []
    log_path = tmp / "gateway.log"
    try:
        for name, port, mode in (("primary", primary, primary_mode), ("secondary", secondary, "ok")):
            mock = ["--name", name, "--port", str(port), "--mode", mode, "--delay", "10"]
            processes.append(
                subprocess.Popen(  # noqa: S603 - our own entry points
                    [sys.executable, "-m", "llm_router.mock_provider", *mock],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            wait_until_up(f"http://127.0.0.1:{port}/_mock/stats")
        router = [
            *("--port", str(gateway), "--db-path", str(tmp / "b.db")),
            *("--primary-url", f"http://127.0.0.1:{primary}/v1", "--primary-model", "p"),
            *("--secondary-url", f"http://127.0.0.1:{secondary}/v1", "--secondary-model", "s"),
            *("--tokens-per-window", str(10**12), "--log-level", log_level),
        ]
        env = {**os.environ, "LLM_ROUTER_TENANT_KEYS": TENANT_KEY}
        with log_path.open("w") as log:
            processes.append(
                subprocess.Popen(  # noqa: S603
                    [sys.executable, "-m", "llm_router", *router],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        wait_until_up(f"http://127.0.0.1:{gateway}/healthz")
        yield f"http://127.0.0.1:{gateway}", f"http://127.0.0.1:{primary}", log_path
    finally:
        for process in processes:
            stop(process)


def stop(process: subprocess.Popen) -> None:
    if sys.platform == "win32":
        # A venv's python.exe is a launcher that runs the real interpreter as a child process, and
        # terminate() would only end the launcher: end the whole tree.
        subprocess.run(  # noqa: S603
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],  # noqa: S607
            capture_output=True,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()


async def drive(url: str, concurrency: int, total: int) -> tuple[list[float], Counter, float]:
    latencies: list[float] = []
    outcomes: Counter = Counter()
    pending = iter(range(total))
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=60, trust_env=False) as client:
        headers = {"Authorization": f"Bearer {TENANT_KEY}"}

        async def worker() -> None:
            for _ in pending:
                t0 = time.perf_counter()
                response = await client.post("/v1/chat/completions", json=BODY, headers=headers)
                latencies.append(time.perf_counter() - t0)
                outcomes[(response.status_code, response.headers.get("x-gateway-route"))] += 1

        started = time.perf_counter()
        async with asyncio.TaskGroup() as tg:
            for _ in range(concurrency):
                tg.create_task(worker())
        return latencies, outcomes, time.perf_counter() - started


def scratch_dir() -> tempfile.TemporaryDirectory:
    # Windows may still hold the killed gateway's database file for a moment.
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def bench_throughput(concurrency: int, total: int, log_level: str, direct: bool) -> None:
    """``direct`` drives the mock primary itself, bypassing the gateway: the rig's own ceiling."""
    with scratch_dir() as tmp, stack(Path(tmp), "ok", log_level) as (gateway_url, primary_url, _):
        url = primary_url if direct else gateway_url
        asyncio.run(drive(url, concurrency, min(total, concurrency * 2)))  # warm up connections
        latencies, outcomes, elapsed = asyncio.run(drive(url, concurrency, total))
    p50, p95, worst = (percentile(latencies, f) * 1e3 for f in (0.5, 0.95, 1.0))
    target = "mock provider directly" if direct else f"gateway, log level {log_level}"
    print(
        f"{total} requests, {concurrency} concurrent, {target}: {total / elapsed:,.0f} req/s; "
        f"latency p50={p50:.0f} ms p95={p95:.0f} ms max={worst:.0f} ms; {dict(outcomes)}"
    )


def bench_failover(levels: list[int]) -> None:
    for concurrency in levels:
        with scratch_dir() as tmp, stack(Path(tmp), "slow", "WARNING") as (url, _, log_path):
            latencies, outcomes, _ = asyncio.run(drive(url, concurrency, concurrency))
            time.sleep(0.5)
            attempts = [
                float(ms)
                for ms in re.findall(
                    r"primary attempt failed: kind=timeout status=None elapsed_ms=(\d+)", log_path.read_text()
                )
            ]
        print(
            f"concurrency {concurrency:>3}: primary attempt cancelled after "
            f"p50={statistics.median(attempts):.0f} ms max={max(attempts):.0f} ms (deadline 3000 ms); "
            f"client latency p50={percentile(latencies, 0.5):.2f} s max={max(latencies):.2f} s; "
            f"{dict(outcomes)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    limiter = sub.add_parser("limiter")
    limiter.add_argument("-n", "--pairs", type=int, default=5_000)
    limiter.add_argument("-c", "--concurrency", type=int, default=400)
    throughput = sub.add_parser("throughput")
    throughput.add_argument("-c", "--concurrency", type=int, default=300)
    throughput.add_argument("-n", "--requests", type=int, default=3_000)
    throughput.add_argument("--log-level", default="INFO")
    throughput.add_argument("--direct", action="store_true", help="drive the mock provider, not the gateway")
    failover = sub.add_parser("failover")
    failover.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1, 100, 200, 400])
    args = parser.parse_args()

    print(
        f"{platform.system()} {platform.release()}, Python {platform.python_version()}, {os.cpu_count()} CPUs"
    )
    if args.command == "limiter":
        bench_limiter(args.pairs, args.concurrency)
    elif args.command == "throughput":
        bench_throughput(args.concurrency, args.requests, args.log_level, args.direct)
    else:
        bench_failover(args.concurrency)


if __name__ == "__main__":
    main()
