"""The scheduler -- the brain.

It runs a single async loop that:
  1. forms a batch from the queue (up to MAX_BATCH_SIZE, or after BATCH_WAIT_MS),
  2. asks the engine to advance every sequence by one token,
  3. streams each token back to its waiting HTTP handler,
  4. retires finished sequences,
  5. (continuous mode) immediately tops the batch back up from the queue so the
     engine never idles -- this is the vLLM / production-serving behaviour.

Static mode instead lets a batch decode to completion before forming the next
one; it's simpler and is what the HFEngine uses.
"""
from __future__ import annotations

import asyncio
import time

from .config import settings
from .metrics import (
    BATCH_SIZE,
    LATENCY,
    RUNNING_SEQS,
    TOKENS_TOTAL,
    TTFT,
    QUEUE_DEPTH,
    stats,
)
from .schemas import DONE, GenerationRequest


class Scheduler:
    def __init__(self, queue, engine) -> None:
        self.queue = queue
        self.engine = engine
        self.running: list[GenerationRequest] = []
        self.continuous = settings.BATCHING == "continuous" and engine.supports_continuous
        self._stop = asyncio.Event()
        self._wait = settings.BATCH_WAIT_MS / 1000.0
        self._max = settings.MAX_BATCH_SIZE

    # -- batch formation ---------------------------------------------------
    def _admit(self) -> None:
        """Pull as many waiting requests as will fit, non-blocking."""
        while len(self.running) < self._max:
            req = self.queue.try_get()
            if req is None:
                break
            req.started_at = time.time()
            self.running.append(req)

    async def _form_initial_batch(self) -> None:
        """Cold start: block for the first request, then briefly let more arrive
        so we don't fire a batch of one when a crowd is milling at the door."""
        first = await self.queue.get()
        first.started_at = time.time()
        self.running.append(first)
        deadline = time.time() + self._wait
        while len(self.running) < self._max and time.time() < deadline:
            self._admit()
            if len(self.running) >= self._max:
                break
            await asyncio.sleep(0.001)

    # -- main loop ---------------------------------------------------------
    async def run(self) -> None:
        while not self._stop.is_set():
            if not self.running:
                await self._form_initial_batch()

            results = await self.engine.step(self.running)

            stats.last_batch = len(self.running)
            stats.running = len(self.running)
            BATCH_SIZE.set(len(self.running))
            RUNNING_SEQS.set(len(self.running))

            finished: list[GenerationRequest] = []
            for req in self.running:
                res = results[req.id]
                if res.token:
                    if req.generated == 0:
                        TTFT.observe(time.time() - req.created_at)
                    req.generated += 1
                    stats.record_token()
                    TOKENS_TOTAL.inc()
                    req.out.put_nowait(res.token)
                if res.finished:
                    latency = time.time() - req.created_at
                    LATENCY.observe(latency)
                    stats.record_completion(latency)
                    req.out.put_nowait(DONE)
                    finished.append(req)

            for req in finished:
                self.running.remove(req)
                self.engine.release(req.id)

            # Continuous batching: refill immediately so the engine never idles.
            if self.continuous:
                self._admit()

            stats.queue_depth = self.queue.depth()
            QUEUE_DEPTH.set(stats.queue_depth)

    def stop(self) -> None:
        self._stop.set()
