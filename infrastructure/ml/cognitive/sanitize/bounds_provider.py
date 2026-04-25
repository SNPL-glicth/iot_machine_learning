"""Bounds providers for the sanitize phase (IMP-1).

Two strategies:
    * :class:`SeriesValuesBoundsProvider` — reads mean±kσ bounds from a
      :class:`SeriesValuesStore` (Redis cross-invocation history).
    * :class:`LocalWindowBoundsProvider` — computes mean±kσ from the
      current input window. Fallback when Redis history is insufficient.

Both implement the informal ``BoundsProvider`` protocol::

    class BoundsProvider(Protocol):
        def get_bounds(
            self,
            series_id: str,
            values: List[float],
            sigma_multiplier: float,
        ) -> Optional[Tuple[float, float]]: ...

Returning ``None`` means "I cannot compute bounds for this request".
"""

from __future__ import annotations

import math
from typing import List, Optional, Protocol, Tuple

from ..series_values import SeriesValuesStore


class BoundsProvider(Protocol):
    """Protocol: return ``(lower, upper)`` or ``None``."""

    def get_bounds(
        self,
        series_id: str,
        values: List[float],
        sigma_multiplier: float,
    ) -> Optional[Tuple[float, float]]:
        ...


class SeriesValuesBoundsProvider:
    """Bounds from the per-series rolling buffer in Redis."""

    def __init__(self, store: SeriesValuesStore, *, min_samples: int = 20) -> None:
        if min_samples <= 0:
            raise ValueError(f"min_samples must be > 0, got {min_samples}")
        self._store = store
        self._min_samples = int(min_samples)

    def get_bounds(
        self,
        series_id: str,
        values: List[float],  # noqa: ARG002 — unused but part of the protocol
        sigma_multiplier: float,
    ) -> Optional[Tuple[float, float]]:
        if not series_id:
            return None
        return self._store.get_bounds(
            series_id,
            sigma_multiplier=sigma_multiplier,
            min_samples=self._min_samples,
        )


class LocalWindowBoundsProvider:
    """Bounds from the current input window only."""

    def __init__(self, *, min_window_size: int = 3) -> None:
        if min_window_size <= 0:
            raise ValueError(f"min_window_size must be > 0, got {min_window_size}")
        self._min_window_size = int(min_window_size)

    def get_bounds(
        self,
        series_id: str,  # noqa: ARG002 — unused
        values: List[float],
        sigma_multiplier: float,
    ) -> Optional[Tuple[float, float]]:
        n = len(values)
        if n < self._min_window_size:
            return None
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        if variance <= 0.0:
            return None
        std = math.sqrt(variance)
        return (mean - sigma_multiplier * std, mean + sigma_multiplier * std)


__all__ = [
    "BoundsProvider",
    "LocalWindowBoundsProvider",
    "SeriesValuesBoundsProvider",
]
