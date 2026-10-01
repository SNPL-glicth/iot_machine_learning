"""Buffer circular de arrepentimiento (Regret Ring Buffer).

Preserva una ventana reciente de observaciones RAW para reevaluación retrospectiva
(backfill) cuando la política conmuta de baja a alta resolución.
"""

from __future__ import annotations

import collections
from typing import Any


class RegretRingBuffer:
    """Buffer circular que preserva las observaciones RAW recientes para backfill."""

    def __init__(self, capacity: int = 30) -> None:
        self.capacity = capacity
        self.buffer: collections.deque[tuple[int, float, float, str]] = collections.deque(
            maxlen=capacity
        )

    def append(self, global_idx: int, val: float, ts_sec: float, ts_raw: str) -> None:
        """Añade un punto crudo al buffer."""
        self.buffer.append((global_idx, val, ts_sec, ts_raw))

    def get_recent_raw_slice(self, n_points: int) -> list[tuple[int, float, float, str]]:
        """Retorna los n puntos más recientes como lista (index, val, ts_sec, ts_raw)."""
        pts = list(self.buffer)
        return pts[-n_points:] if len(pts) >= n_points else pts

    @property
    def memory_bytes(self) -> int:
        """Estimación aproximada de memoria ocupada en bytes."""
        return self.capacity * 64
