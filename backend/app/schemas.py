"""Wire schemas (what crosses HTTP) and the internal GenerationRequest that
flows through queue -> scheduler -> engine."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum

from pydantic import BaseModel, Field


class Priority(IntEnum):
    """Lower value = served first. Backs the PriorityQueue ordering."""
    PREMIUM = 0
    NORMAL = 1
    FREE = 2


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    tier: str = Field(default="normal")  # premium | normal | free


# Sentinel pushed onto a request's output queue to signal completion.
DONE = object()


@dataclass(order=True)
class GenerationRequest:
    """One in-flight generation. `order` fields make it sortable inside the
    PriorityQueue: first by tier, then by arrival (FIFO within a tier)."""
    priority: int
    seq: int
    # Everything below is excluded from ordering comparisons.
    id: str = field(compare=False, default="")
    prompt: str = field(compare=False, default="")
    max_tokens: int = field(compare=False, default=64)
    created_at: float = field(compare=False, default_factory=time.time)
    started_at: float | None = field(compare=False, default=None)
    out: asyncio.Queue = field(compare=False, default_factory=asyncio.Queue)
    generated: int = field(compare=False, default=0)
