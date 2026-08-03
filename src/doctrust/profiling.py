from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def gpu_measurement() -> Iterator[dict[str, float | None]]:
    """Measure wall time and peak CUDA allocation for one inference call."""
    stats: dict[str, float | None] = {"latency_seconds": None, "peak_vram_mb": None}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        yield stats
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            stats["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        stats["latency_seconds"] = time.perf_counter() - start
