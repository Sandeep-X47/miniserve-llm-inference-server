"""The request queue. A bounded asyncio.PriorityQueue gives us two systems
concepts for free: priority scheduling (premium > normal > free) and
backpressure (reject when full instead of letting memory explode)."""
from __future__ import annotations

import asyncio
import itertools

from .config import settings
from .schemas import GenerationRequest


class QueueFull(Exception):
    """Raised when the queue is at capacity. The API turns this into HTTP 429."""


class RequestQueue:
    def __init__(self, capacity: int = settings.QUEUE_CAPACITY) -> None:
        self._q: asyncio.PriorityQueue[GenerationRequest] = asyncio.PriorityQueue(
            maxsize=capacity
        )
        self._seq = itertools.count()

    def next_seq(self) -> int:
        return next(self._seq)

    def submit(self, req: GenerationRequest) -> None:
        """Non-blocking enqueue. Backpressure: refuse rather than buffer forever."""
        try:
            self._q.put_nowait(req)
        except asyncio.QueueFull as exc:
            raise QueueFull() from exc

    def try_get(self) -> GenerationRequest | None:
        try:
            return self._q.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get(self) -> GenerationRequest:
        """Block until a request is available. Used on scheduler cold start."""
        return await self._q.get()

    def depth(self) -> int:
        return self._q.qsize()
