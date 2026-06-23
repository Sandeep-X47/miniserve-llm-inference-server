"""Inference engines.

The scheduler is engine-agnostic: it hands the engine the current batch of
GenerationRequests each iteration and gets back one token per request. The
engine owns the decoding mechanics (canned tokens, or a real transformer
forward pass with a KV cache).

Two implementations:
  MockEngine - simulates a transformer. Constant per-step latency regardless of
               batch size, which is exactly why batching raises throughput. No
               downloads, runs on any machine. Supports continuous batching.
  HFEngine   - a real Hugging Face causal LM with a real KV cache. Static
               batching only (a batch decodes to completion before the next
               forms); correct and demonstrable, just not vLLM-grade.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .config import settings
from .schemas import GenerationRequest


@dataclass
class StepResult:
    token: str
    finished: bool


# A small, deterministic, on-topic-ish corpus so the mock demo reads as real
# text rather than gibberish. Real responses come from HFEngine.
_TEMPLATE = (
    "Here is a concise explanation. {topic} works by coordinating independent "
    "components through a well defined protocol, so that the overall system "
    "stays correct even when individual parts are slow or fail. The key idea is "
    "to batch related work, keep shared state minimal, and make progress "
    "continuously rather than in large stop the world steps. In practice this "
    "yields higher throughput and steadier latency under load."
).split()


class MockEngine:
    supports_continuous = True

    def __init__(self) -> None:
        self._state: dict[str, dict] = {}
        self._latency = settings.MOCK_STEP_LATENCY_MS / 1000.0

    def _tokens_for(self, req: GenerationRequest) -> list[str]:
        topic = req.prompt.strip().rstrip("?.").split("\n")[0][:60] or "the system"
        body = _TEMPLATE.copy()
        body[body.index("{topic}")] = topic  # fill the placeholder
        # Make different prompts diverge a little in length.
        repeat = 1 + (len(req.prompt) % 3)
        return (body * repeat)[: req.max_tokens]

    async def step(self, batch: list[GenerationRequest]) -> dict[str, StepResult]:
        # One simulated forward pass. Cost is independent of batch size: that is
        # the whole point -- a GPU processes 1 or 64 sequences in ~the same time.
        await asyncio.sleep(self._latency)

        results: dict[str, StepResult] = {}
        for req in batch:
            st = self._state.get(req.id)
            if st is None:
                st = {"tokens": self._tokens_for(req), "cursor": 0}
                self._state[req.id] = st
            cursor = st["cursor"]
            toks = st["tokens"]
            if cursor >= len(toks) or req.generated >= req.max_tokens:
                results[req.id] = StepResult("", finished=True)
                continue
            tok = toks[cursor]
            st["cursor"] = cursor + 1
            piece = tok if cursor == 0 else " " + tok
            results[req.id] = StepResult(piece, finished=False)
        return results

    def release(self, req_id: str) -> None:
        self._state.pop(req_id, None)


class HFEngine:
    """Real transformer with a real KV cache. Static batching.

    Not exercised in the no-GPU sandbox (weights need a download), but written
    to run as-is on a machine with internet and torch installed:
        ENGINE=hf HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct DEVICE=cuda uvicorn ...
    """

    supports_continuous = False

    def __init__(self) -> None:
        import torch  # noqa: lazy import so mock mode needs no torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = settings.DEVICE
        self.tok = AutoTokenizer.from_pretrained(settings.HF_MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"  # required for batched left-pad decoding
        self.model = AutoModelForCausalLM.from_pretrained(settings.HF_MODEL).to(
            self.device
        )
        self.model.eval()
        self._batch_key: tuple[str, ...] = ()
        self._past = None
        self._last: "torch.Tensor | None" = None
        self._attn: "torch.Tensor | None" = None
        self._order: list[str] = []

    def _prefill(self, batch: list[GenerationRequest]) -> dict[str, StepResult]:
        torch = self.torch
        prompts = [b.prompt for b in batch]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, use_cache=True)
        self._past = out.past_key_values
        self._attn = enc["attention_mask"]
        self._order = [b.id for b in batch]
        self._batch_key = tuple(self._order)
        next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        self._last = next_ids
        # Grow attention mask for the freshly produced token.
        self._attn = torch.cat(
            [self._attn, torch.ones_like(next_ids)], dim=1
        )
        return self._decode_results(batch, next_ids)

    def _decode_step(self, batch: list[GenerationRequest]) -> dict[str, StepResult]:
        torch = self.torch
        with torch.no_grad():
            out = self.model(
                input_ids=self._last,
                past_key_values=self._past,
                attention_mask=self._attn,
                use_cache=True,
            )
        self._past = out.past_key_values
        next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        self._last = next_ids
        self._attn = torch.cat([self._attn, torch.ones_like(next_ids)], dim=1)
        return self._decode_results(batch, next_ids)

    def _decode_results(self, batch, next_ids) -> dict[str, StepResult]:
        results: dict[str, StepResult] = {}
        eos = self.tok.eos_token_id
        for row, req in enumerate(batch):
            tid = int(next_ids[row, 0])
            finished = tid == eos or req.generated >= req.max_tokens
            text = "" if finished else self.tok.decode([tid])
            results[req.id] = StepResult(text, finished=finished)
        return results

    async def step(self, batch: list[GenerationRequest]) -> dict[str, StepResult]:
        # Run the blocking forward pass off the event loop.
        key = tuple(b.id for b in batch)
        fn = self._prefill if key != self._batch_key else self._decode_step
        return await asyncio.to_thread(fn, batch)

    def release(self, req_id: str) -> None:
        # Static batching tears the whole batch down at once when it empties.
        if all(i == req_id for i in self._batch_key) or len(self._batch_key) <= 1:
            self._past = None
            self._batch_key = ()


def build_engine():
    if settings.ENGINE == "hf":
        return HFEngine()
    return MockEngine()
