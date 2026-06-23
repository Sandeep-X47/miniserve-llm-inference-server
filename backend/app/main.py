"""HTTP surface.

  POST /chat     -> Server-Sent Events stream of tokens (or 429 under backpressure)
  GET  /stats    -> JSON snapshot for the admin dashboard
  GET  /metrics  -> Prometheus scrape endpoint
  GET  /health   -> liveness + engine/config info
"""
from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import settings
from .engine import build_engine
from .metrics import REQUESTS_REJECTED, REQUESTS_TOTAL, stats
from .queue_manager import QueueFull, RequestQueue
from .scheduler import Scheduler
from .schemas import DONE, ChatRequest, GenerationRequest, Priority

app = FastAPI(title="Mini LLM Inference Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

queue = RequestQueue()
engine = build_engine()
scheduler = Scheduler(queue, engine)

_TIER = {"premium": Priority.PREMIUM, "normal": Priority.NORMAL, "free": Priority.FREE}


@app.on_event("startup")
async def _startup() -> None:
    app.state.task = asyncio.create_task(scheduler.run())


@app.on_event("shutdown")
async def _shutdown() -> None:
    scheduler.stop()
    app.state.task.cancel()


@app.post("/chat")
async def chat(body: ChatRequest):
    priority = _TIER.get(body.tier.lower(), Priority.NORMAL)
    max_tokens = min(
        body.max_tokens or settings.DEFAULT_MAX_TOKENS, settings.HARD_MAX_TOKENS
    )
    req = GenerationRequest(
        priority=int(priority),
        seq=queue.next_seq(),
        id=uuid.uuid4().hex[:12],
        prompt=body.prompt,
        max_tokens=max_tokens,
    )

    try:
        queue.submit(req)
    except QueueFull:
        REQUESTS_REJECTED.inc()
        stats.rejected += 1
        return JSONResponse(
            status_code=429,
            content={
                "error": "server at capacity",
                "detail": "request queue is full; retry with backoff",
            },
            headers={"Retry-After": "1"},
        )

    REQUESTS_TOTAL.labels(tier=body.tier.lower()).inc()
    stats.record_request()

    async def event_stream():
        # SSE frames. The scheduler pushes tokens onto req.out; we relay them.
        yield f"data: {json.dumps({'id': req.id, 'event': 'start'})}\n\n"
        while True:
            item = await req.out.get()
            if item is DONE:
                yield f"data: {json.dumps({'event': 'done'})}\n\n"
                break
            yield f"data: {json.dumps({'event': 'token', 'text': item})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stats")
async def get_stats():
    snap = stats.snapshot()
    snap["queue_capacity"] = settings.QUEUE_CAPACITY
    snap["max_batch_size"] = settings.MAX_BATCH_SIZE
    snap["batching"] = "continuous" if scheduler.continuous else "static"
    snap["engine"] = settings.ENGINE
    return snap


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": settings.ENGINE,
        "model": settings.HF_MODEL if settings.ENGINE == "hf" else "mock",
        "batching": "continuous" if scheduler.continuous else "static",
        "max_batch_size": settings.MAX_BATCH_SIZE,
        "queue_capacity": settings.QUEUE_CAPACITY,
    }


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
