"""Benchmark: throughput vs. batch size -- the centerpiece graph.

Runs the *real* scheduler + engine in-process (no network noise) at a range of
batch sizes and measures aggregate tokens/sec. With the MockEngine, per-step
latency is constant regardless of batch size, so throughput scales with the
batch -- which is precisely the property real GPUs have and the reason dynamic
batching exists.

    python benchmark.py                  # mock engine, writes benchmark.png
    ENGINE=hf python benchmark.py        # real model numbers (needs GPU/weights)

IMPORTANT: with the mock engine these are *simulated* numbers that illustrate
the scaling law. Absolute tokens/sec on real hardware will differ; the shape of
the curve is the point. Label it honestly on your resume.
"""
from __future__ import annotations

import asyncio
import os
import time

# Allow `python benchmark.py` from the backend/ dir.
import sys

sys.path.insert(0, os.path.dirname(__file__))

from app import config  # noqa: E402

REQUESTS = int(os.getenv("BENCH_REQUESTS", "96"))
MAX_TOKENS = int(os.getenv("BENCH_MAX_TOKENS", "48"))
BATCH_SIZES = [int(b) for b in os.getenv("BENCH_BATCHES", "1,2,4,8,16,32").split(",")]


async def run_one(batch_size: int) -> tuple[float, int]:
    # Rebuild the world for a clean measurement at this batch size.
    config.settings.MAX_BATCH_SIZE = batch_size
    config.settings.BATCHING = "continuous"

    from app.engine import build_engine
    from app.queue_manager import RequestQueue
    from app.scheduler import Scheduler
    from app.schemas import DONE, GenerationRequest

    queue = RequestQueue(capacity=10_000)
    engine = build_engine()
    sched = Scheduler(queue, engine)

    reqs = []
    for i in range(REQUESTS):
        r = GenerationRequest(
            priority=1, seq=queue.next_seq(), id=f"r{i}",
            prompt=f"Explain distributed system number {i}", max_tokens=MAX_TOKENS,
        )
        queue.submit(r)
        reqs.append(r)

    async def drain(r):
        tokens = 0
        while True:
            item = await r.out.get()
            if item is DONE:
                return tokens
            tokens += 1

    task = asyncio.create_task(sched.run())
    start = time.time()
    counts = await asyncio.gather(*(drain(r) for r in reqs))
    elapsed = time.time() - start
    sched.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    total_tokens = sum(counts)
    return total_tokens / elapsed, total_tokens


async def main() -> None:
    print(f"engine={config.settings.ENGINE}  requests={REQUESTS}  max_tokens={MAX_TOKENS}")
    print(f"{'batch':>6} | {'tokens/sec':>12} | {'speedup':>8}")
    print("-" * 34)
    results = []
    baseline = None
    for b in BATCH_SIZES:
        tps, total = await run_one(b)
        if baseline is None:
            baseline = tps
        results.append((b, tps))
        print(f"{b:>6} | {tps:>12.1f} | {tps / baseline:>7.2f}x")

    _plot(results)


def _plot(results) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping graph. `pip install matplotlib`)")
        return

    labels = [f"batch={b}" for b, _ in results]
    values = [round(t) for _, t in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color="#2b2b2b")
    bars[-1].set_color("#e8483f")  # highlight the best
    ax.set_ylabel("throughput (tokens / sec)")
    ax.set_title("Dynamic batching: throughput vs. batch size")
    speedup = values[-1] / values[0] if values[0] else 0
    ax.text(
        0.02, 0.95, f"{speedup:.1f}x  vs. one-at-a-time",
        transform=ax.transAxes, va="top", fontsize=12, weight="bold", color="#e8483f",
    )
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                ha="center", va="bottom", fontsize=9)
    note = "Simulated (mock engine)" if config.settings.ENGINE != "hf" else "Real model"
    ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right",
            fontsize=8, color="#888", style="italic")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "benchmark.png")
    fig.savefig(out, dpi=130)
    print(f"\nGraph written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
