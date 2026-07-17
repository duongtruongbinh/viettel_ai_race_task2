"""GPU selection with a polling wait gate (never abort on a busy GPU).

Heavy steps (KB embedding, encoder training, LLM hosting) call
:func:`wait_for_gpu`; if no device has enough free VRAM it sleeps and retries up
to a ceiling, logging each check, rather than failing.  CPU-only steps must not
call this.
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import List, Optional

log = logging.getLogger("medextract.gpu")


def _query_free_mb() -> List[int]:
    """Return free VRAM (MB) per GPU via nvidia-smi; [] if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def pick_device(min_free_gb: float = 4.0) -> Optional[int]:
    """Return the GPU id with the most free VRAM ≥ min_free_gb, else None."""
    free = _query_free_mb()
    if not free:
        return None
    best = max(range(len(free)), key=lambda i: free[i])
    return best if free[best] >= min_free_gb * 1024 else None


def wait_for_gpu(
    min_free_gb: float = 4.0,
    poll_seconds: int = 300,
    max_wait_hours: float = 12.0,
) -> Optional[int]:
    """Block until a GPU with ≥ min_free_gb free VRAM appears; return its id.

    Returns None only if no CUDA GPUs exist at all (caller falls back to CPU).
    Raises TimeoutError after max_wait_hours.
    """
    if not _query_free_mb():
        log.info("[gpu-wait] no CUDA GPU detected; using CPU")
        return None

    deadline = time.monotonic() + max_wait_hours * 3600
    while True:
        dev = pick_device(min_free_gb)
        if dev is not None:
            free = _query_free_mb()[dev]
            log.info("[gpu-wait] using cuda:%d (%.1f GB free)", dev, free / 1024)
            return dev
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"no GPU freed ≥{min_free_gb} GB within {max_wait_hours}h"
            )
        log.info(
            "[gpu-wait] no device with ≥%.1f GB free, retrying in %ds…",
            min_free_gb, poll_seconds,
        )
        time.sleep(poll_seconds)


def resolve_device(device: str = "auto", min_free_gb: float = 4.0) -> str:
    """Resolve a config device string to a torch device string.

    ``"auto"`` picks a free GPU if one exists right now, else CPU (no waiting).
    ``"wait"`` blocks via :func:`wait_for_gpu`.  ``"cpu"``/``"cuda:N"`` pass through.
    """
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda"):
        return device
    if device == "wait":
        d = wait_for_gpu(min_free_gb=min_free_gb)
        return f"cuda:{d}" if d is not None else "cpu"
    # auto
    d = pick_device(min_free_gb)
    return f"cuda:{d}" if d is not None else "cpu"
