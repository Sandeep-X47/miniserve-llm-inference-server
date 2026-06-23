"""Observability. Prometheus for the real scrape endpoint, plus a cheap
rolling snapshot the admin dashboard polls once a second."""
from __future__ import annotations

import time
from collections import deque

from prometheus_client import Counter, Gauge, Histogram

# --- Prometheus instruments ----------------------------------------------
REQUESTS_TOTAL = Counter("infer_requests_total", "Requests accepted", ["tier"])
REQUESTS_REJECTED = Counter("infer_requests_rejected_total", "Rejected (backpressure)")
TOKENS_TOTAL = Counter("infer_tokens_total", "Tokens generated")
QUEUE_DEPTH = Gauge("infer_queue_depth", "Requests waiting in queue")
RUNNING_SEQS = Gauge("infer_running_sequences", "Sequences in the active batch")
BATCH_SIZE = Gauge("infer_batch_size", "Size of the most recent step batch")
LATENCY = Histogram(
    "infer_request_latency_seconds",
    "End-to-end latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
TTFT = Histogram(
    "infer_time_to_first_token_seconds",
    "Time to first token",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


class LiveStats:
    """Rolling counters for the dashboard. Not authoritative -- Prometheus is.
    This just makes a pretty JSON without a Prometheus query layer."""

    def __init__(self) -> None:
        self._token_times: deque[float] = deque(maxlen=4000)
        self._req_times: deque[float] = deque(maxlen=2000)
        self.queue_depth = 0
        self.running = 0
        self.last_batch = 0
        self.accepted = 0
        self.rejected = 0
        self.completed = 0
        self.tokens = 0
        self._latencies: deque[float] = deque(maxlen=200)

    def record_token(self) -> None:
        self._token_times.append(time.time())
        self.tokens += 1

    def record_request(self) -> None:
        self._req_times.append(time.time())
        self.accepted += 1

    def record_completion(self, latency: float) -> None:
        self.completed += 1
        self._latencies.append(latency)

    def _rate(self, times: deque[float], window: float = 5.0) -> float:
        now = time.time()
        recent = [t for t in times if now - t <= window]
        return len(recent) / window if recent else 0.0

    def snapshot(self) -> dict:
        lats = sorted(self._latencies)
        p50 = lats[len(lats) // 2] if lats else 0.0
        p95 = lats[int(len(lats) * 0.95)] if lats else 0.0
        return {
            "queue_depth": self.queue_depth,
            "running_sequences": self.running,
            "last_batch_size": self.last_batch,
            "tokens_per_sec": round(self._rate(self._token_times), 1),
            "requests_per_sec": round(self._rate(self._req_times), 2),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "completed": self.completed,
            "tokens_total": self.tokens,
            "latency_p50_ms": round(p50 * 1000, 1),
            "latency_p95_ms": round(p95 * 1000, 1),
        }


stats = LiveStats()
