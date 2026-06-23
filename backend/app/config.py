"""Central configuration. Everything is overridable via environment variables
so the same code runs as a laptop mock or a GPU deployment with no edits."""
from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


class Settings:
    # --- Engine selection -------------------------------------------------
    # "mock" : zero-download simulator, runs anywhere, default.
    # "hf"   : real Hugging Face transformers model (needs internet + weights).
    ENGINE: str = os.getenv("ENGINE", "mock")
    HF_MODEL: str = os.getenv("HF_MODEL", "sshleifer/tiny-gpt2")
    DEVICE: str = os.getenv("DEVICE", "cpu")  # "cuda" if you have a GPU

    # --- Scheduler --------------------------------------------------------
    MAX_BATCH_SIZE: int = _int("MAX_BATCH_SIZE", 16)
    # Max wall-clock the scheduler waits to fill a batch before firing anyway.
    BATCH_WAIT_MS: float = _float("BATCH_WAIT_MS", 15.0)
    # "continuous" = requests join the running batch mid-generation (vLLM style).
    # "static"     = a batch runs to completion before the next one forms.
    BATCHING: str = os.getenv("BATCHING", "continuous")

    # --- Backpressure -----------------------------------------------------
    QUEUE_CAPACITY: int = _int("QUEUE_CAPACITY", 256)

    # --- Generation -------------------------------------------------------
    DEFAULT_MAX_TOKENS: int = _int("DEFAULT_MAX_TOKENS", 64)
    HARD_MAX_TOKENS: int = _int("HARD_MAX_TOKENS", 512)

    # --- Mock engine timing ----------------------------------------------
    # One simulated forward step. In a real GPU this cost is ~constant whether
    # the batch holds 1 or 64 sequences -- that constancy is *why* batching wins.
    MOCK_STEP_LATENCY_MS: float = _float("MOCK_STEP_LATENCY_MS", 25.0)

    # --- CORS -------------------------------------------------------------
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")


settings = Settings()
