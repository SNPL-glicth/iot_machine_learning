"""Canonical in-memory sliding window store with LRU + TTL eviction (E-15).

This is the single canonical implementation that unifies the patterns from:
- SlidingWindowStore (ml_service/consumers/sliding_window.py)
- SlidingWindowBuffer (ml_service/sliding_window_buffer.py)

Both existing implementations remain functional (backward compat).
New code should use this canonical version via ISlidingWindowStore port.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Generic, List, Tuple, TypeVar

from iot_machine_learning.domain.ports.sliding_window_port import (
    ISlidingWindowStore,
    WindowConfig,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _SensorEntry(Generic[T]):
    """Internal entry wrapping a deque with last-access monotonic time."""

    items: Deque[Tuple[float, T]]
    last_accessed: float = field(default_factory=time.monotonic)


class InMemorySlidingWindowStore(ISlidingWindowStore[T]):
    """Canonical in-memory per-sensor sliding window store.

    Features:
    - Thread-safe via ``threading.Lock``
    - FIFO eviction by ``max_size`` per sensor
    - LRU eviction when ``max_sensors`` reached
    - TTL eviction for idle sensors
    - Metrics for monitoring
    """

    def __init__(self, config: WindowConfig | None = None) -> None:
        self._config = config or WindowConfig()
        self._entries: OrderedDict[int, _SensorEntry[T]] = OrderedDict()
        self._lock = threading.Lock()
        self._evictions_lru: int = 0
        self._evictions_ttl: int = 0

    # ------------------------------------------------------------------
    # ISlidingWindowStore implementation
    # ------------------------------------------------------------------

    def append(self, sensor_id: int, item: T, timestamp: float) -> int:
        now = time.monotonic()
        with self._lock:
            self._cleanup_expired(now)
            if sensor_id in self._entries:
                entry = self._entries[sensor_id]
                self._entries.move_to_end(sensor_id)
            else:
                self._evict_lru_if_full()
                entry = _SensorEntry(
                    items=deque(maxlen=self._config.max_size),
                    last_accessed=now,
                )
                self._entries[sensor_id] = entry
            entry.items.append((timestamp, item))
            entry.last_accessed = now
            return len(entry.items)

    def get_window(self, sensor_id: int) -> List[T]:
        with self._lock:
            entry = self._entries.get(sensor_id)
            if entry is None:
                return []
            self._entries.move_to_end(sensor_id)
            entry.last_accessed = time.monotonic()
            items = list(entry.items)
        items.sort(key=lambda pair: pair[0])
        return [item for _, item in items]

    def get_size(self, sensor_id: int) -> int:
        with self._lock:
            entry = self._entries.get(sensor_id)
            return len(entry.items) if entry else 0

    def clear(self, sensor_id: int) -> None:
        with self._lock:
            entry = self._entries.get(sensor_id)
            if entry is not None:
                entry.items.clear()

    def sensor_ids(self) -> List[int]:
        with self._lock:
            return list(self._entries.keys())

    def evict_stale(self, current_time: float) -> int:
        with self._lock:
            return self._cleanup_expired(current_time)

    def get_metrics(self) -> dict:
        with self._lock:
            return {
                "active_sensors": len(self._entries),
                "max_sensors": self._config.max_sensors,
                "max_size_per_sensor": self._config.max_size,
                "evictions_lru": self._evictions_lru,
                "evictions_ttl": self._evictions_ttl,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_lru_if_full(self) -> None:
        """Pop oldest entry when at capacity (caller holds lock)."""
        while len(self._entries) >= self._config.max_sensors:
            evicted_id, _ = self._entries.popitem(last=False)
            self._evictions_lru += 1
            logger.debug("canonical_lru_evict sensor_id=%d", evicted_id)

    def _cleanup_expired(self, now: float) -> int:
        """Remove entries idle longer than TTL (caller holds lock)."""
        cutoff = now - self._config.ttl_seconds
        expired = [
            sid for sid, e in self._entries.items()
            if e.last_accessed < cutoff
        ]
        for sid in expired:
            del self._entries[sid]
            self._evictions_ttl += 1
            logger.debug("canonical_ttl_evict sensor_id=%d", sid)
        return len(expired)
