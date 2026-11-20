"""Latency of GET /companies/{id}/similar against a running API, measured from the client.

Usage: uv run python -m scripts.loadtest [--url http://127.0.0.1:8000] [--requests 2000] [--concurrency 8]
"""

import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import psycopg

from similar.config import load_settings


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    with psycopg.connect(load_settings().database_url) as conn:
        ids = [r[0] for r in conn.execute("SELECT registration_number FROM companies WHERE status = 'active'")]
    rng = random.Random(0)
    targets = [rng.choice(ids) for _ in range(args.requests)]

    with httpx.Client(base_url=args.url, timeout=10) as client:
        for reg in targets[:50]:  # warm-up: connection pool, plan cache
            client.get(f"/companies/{reg}/similar")

        def one(reg: str) -> tuple[float, int]:
            start = time.perf_counter()
            status = client.get(f"/companies/{reg}/similar").status_code
            return (time.perf_counter() - start) * 1000, status

        start = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as ex:
            results = list(ex.map(one, targets))
        wall = time.perf_counter() - start

    ms = [r[0] for r in results]
    statuses = sorted({r[1] for r in results})
    print(f"requests={len(ms)} concurrency={args.concurrency} statuses={statuses} throughput={len(ms) / wall:.0f} req/s")
    print(f"p50={percentile(ms, 50):.1f}ms p95={percentile(ms, 95):.1f}ms p99={percentile(ms, 99):.1f}ms "
          f"max={max(ms):.1f}ms mean={statistics.mean(ms):.1f}ms")


if __name__ == "__main__":
    main()
