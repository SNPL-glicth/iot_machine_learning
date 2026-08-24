"""Market Replay Engine Degraded Windows (FASE 6)."""

from __future__ import annotations

from .config import ReplayEngineConfig


def in_degraded_window(timestamp: float, cfg: ReplayEngineConfig) -> bool:
    """True si ``timestamp`` cae en una ventana degradada (FASE 6)."""
    return any(
        start <= timestamp <= end
        for start, end in cfg.degraded_windows
    )