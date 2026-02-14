"""Sliding window buffer with LRU + TTL eviction (RC-3 fix)."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Deque, Dict, Iterable, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SENSORS: int = 1000
_DEFAULT_TTL_SECONDS: float = 3600.0


@dataclass(frozen=True)
class WindowStats:
    """Estadísticos agregados para una ventana de tiempo."""

    window_seconds: float
    mean: float
    min: float
    max: float
    std_dev: float
    trend: float  # pendiente aproximada valor/segundo
    count: int
    last_value: float


@dataclass
class _BufferEntry:
    """Internal entry wrapping a deque with last-access monotonic time."""
    buf: Deque[Tuple[float, float]]
    last_accessed: float = field(default_factory=time.monotonic)


class SlidingWindowBuffer:
    """Buffer deslizante en memoria por sensor con LRU + TTL eviction."""

    def __init__(
        self,
        max_horizon_seconds: float = 10.0,
        max_sensors: int = _DEFAULT_MAX_SENSORS,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._max_horizon_seconds = float(max_horizon_seconds)
        self._max_sensors = max(1, max_sensors)
        self._ttl = float(ttl_seconds)
        self._entries: OrderedDict[int, _BufferEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._evictions_lru: int = 0
        self._evictions_ttl: int = 0

    def add_reading(
        self,
        sensor_id: int,
        value: float,
        timestamp: float,
        windows: Iterable[float] = (1.0, 5.0, 10.0),
    ) -> Dict[str, WindowStats]:
        """Añade una lectura y devuelve stats por ventana."""
        now = time.monotonic()
        with self._lock:
            self._cleanup_expired(now)
            if sensor_id in self._entries:
                entry = self._entries[sensor_id]
                self._entries.move_to_end(sensor_id)
            else:
                self._evict_lru_if_full()
                entry = _BufferEntry(buf=deque(), last_accessed=now)
                self._entries[sensor_id] = entry

            entry.buf.append((timestamp, float(value)))
            entry.last_accessed = now

            # Recortar lecturas fuera del horizonte máximo
            cutoff = timestamp - self._max_horizon_seconds
            while entry.buf and entry.buf[0][0] < cutoff:
                entry.buf.popleft()

            return self._compute_stats(entry.buf, timestamp, windows)

    def _compute_stats(
        self,
        buf: Deque[Tuple[float, float]],
        now_ts: float,
        windows: Iterable[float],
    ) -> Dict[str, WindowStats]:
        result: Dict[str, WindowStats] = {}
        if not buf:
            return result

        for w in windows:
            w = float(w)
            cutoff = now_ts - w
            window_points = [item for item in buf if item[0] >= cutoff]
            if not window_points:
                continue

            values = [v for (_, v) in window_points]
            v_min = min(values)
            v_max = max(values)
            v_mean = sum(values) / len(values)

            if len(values) >= 2:
                mean_diff_sq = [(v - v_mean) ** 2 for v in values]
                std_dev = sqrt(sum(mean_diff_sq) / len(values))
            else:
                std_dev = 0.0

            if len(window_points) >= 2:
                t0, v0 = window_points[0]
                t1, v1 = window_points[-1]
                dt = max(t1 - t0, 1e-3)  # evitar división por 0
                trend = (v1 - v0) / dt
            else:
                trend = 0.0

            key = f"w{int(w)}"
            result[key] = WindowStats(
                window_seconds=w,
                mean=v_mean,
                min=v_min,
                max=v_max,
                std_dev=std_dev,
                trend=trend,
                count=len(values),
                last_value=values[-1],
            )

        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        """Return eviction / occupancy metrics for monitoring."""
        with self._lock:
            return {
                "active_sensors": len(self._entries),
                "max_sensors": self._max_sensors,
                "evictions_lru": self._evictions_lru,
                "evictions_ttl": self._evictions_ttl,
            }

    # ------------------------------------------------------------------
    # Internal eviction
    # ------------------------------------------------------------------

    def _evict_lru_if_full(self) -> None:
        """Pop oldest entry when at capacity (caller holds lock)."""
        while len(self._entries) >= self._max_sensors:
            evicted_id, _ = self._entries.popitem(last=False)
            self._evictions_lru += 1
            logger.debug("buffer_lru_evict sensor_id=%d", evicted_id)

    def _cleanup_expired(self, now: float) -> None:
        """Remove entries idle longer than TTL (caller holds lock)."""
        cutoff = now - self._ttl
        expired = [
            sid for sid, e in self._entries.items()
            if e.last_accessed < cutoff
        ]
        for sid in expired:
            del self._entries[sid]
            self._evictions_ttl += 1
            logger.debug("buffer_ttl_evict sensor_id=%d", sid)
