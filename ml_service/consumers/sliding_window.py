"""In-memory sliding window store per sensor_id with LRU + TTL eviction.

Fixes RC-3 (memory leak): inactive sensors are now evicted automatically.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SENSORS: int = 1000
_DEFAULT_TTL_SECONDS: float = 3600.0
_DEFAULT_CLEANUP_ENABLED: bool = True


@dataclass
class Reading:
    """A single sensor reading from the stream."""
    sensor_id: int
    value: float
    timestamp: float
    timestamp_iso: str


@dataclass
class _WindowEntry:
    """Internal entry wrapping a deque with last-access monotonic time."""
    window: deque
    last_accessed: float = field(default_factory=time.monotonic)


class SlidingWindowStore:
    """LRU + TTL cache of sliding windows per sensor_id.

    - **LRU**: oldest-accessed sensor evicted when ``max_sensors`` reached.
    - **TTL**: sensors idle longer than ``ttl_seconds`` are purged lazily.
    - Thread-safe via ``threading.Lock``.
    - Public API unchanged: ``append``, ``get_window``, ``clear``, ``sensor_ids``.
    """

    def __init__(
        self,
        max_size: int = 20,
        max_sensors: int = _DEFAULT_MAX_SENSORS,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        enable_proactive_cleanup: bool = _DEFAULT_CLEANUP_ENABLED,
        max_total_entries: int = 50000,
        flush_callback: Optional[Callable[[int, List[Reading]], None]] = None,
    ) -> None:
        self._max_size = max_size
        self._max_sensors = max(1, max_sensors)
        self._max_total_entries = max(1, max_total_entries)
        self._ttl = float(ttl_seconds)
        self._entries: OrderedDict[int, _WindowEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._evictions_lru: int = 0
        self._evictions_ttl: int = 0
        self._evictions_global: int = 0
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        
        # Flush callback: invoked before eviction to persist data
        self._flush_callback = flush_callback
        
        if enable_proactive_cleanup and ttl_seconds > 0:
            self._cleanup_thread = threading.Thread(
                target=self._periodic_cleanup,
                daemon=True,
                name="SlidingWindowCleanup"
            )
            self._cleanup_thread.start()
            atexit.register(self.close)

    # ------------------------------------------------------------------
    # Public API (backward-compatible)
    # ------------------------------------------------------------------

    def append(self, reading: Reading) -> int:
        """Append reading and return current window size."""
        sid = reading.sensor_id
        now = time.monotonic()
        with self._lock:
            self._cleanup_expired(now)
            if sid in self._entries:
                entry = self._entries[sid]
                self._entries.move_to_end(sid)
            else:
                self._evict_lru_if_full()
                entry = _WindowEntry(
                    window=deque(maxlen=self._max_size), last_accessed=now,
                )
                self._entries[sid] = entry
            entry.window.append(reading)
            entry.last_accessed = now
            self._evict_global_if_needed()
            return len(entry.window)

    def get_window(self, sensor_id: int) -> List[Reading]:
        """Return readings sorted by timestamp (chronological)."""
        with self._lock:
            entry = self._entries.get(sensor_id)
            if entry is None:
                return []
            self._entries.move_to_end(sensor_id)
            entry.last_accessed = time.monotonic()
            readings = list(entry.window)
        readings.sort(key=lambda r: r.timestamp)
        return readings

    def clear(self, sensor_id: int) -> None:
        with self._lock:
            entry = self._entries.get(sensor_id)
            if entry is not None:
                entry.window.clear()

    @property
    def sensor_ids(self) -> List[int]:
        with self._lock:
            return list(self._entries.keys())

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
                "evictions_global": self._evictions_global,
                "total_entries": self._get_total_entries(),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_lru_if_full(self) -> None:
        """Pop oldest entry when at capacity (caller holds lock).
        
        Flushes data to callback before eviction if configured.
        """
        while len(self._entries) >= self._max_sensors:
            evicted_id, entry = self._entries.popitem(last=False)
            
            # Flush before eviction to prevent data loss
            if self._flush_callback is not None:
                try:
                    readings = list(entry.window)
                    if readings:
                        self._flush_callback(evicted_id, readings)
                except Exception as e:
                    logger.warning(f"flush_failed sensor_id={evicted_id}: {e}")
            
            self._evictions_lru += 1
            logger.debug("lru_evict sensor_id=%d", evicted_id)

    def _get_total_entries(self) -> int:
        """Count total entries across all sensors (caller holds lock)."""
        return sum(len(entry.window) for entry in self._entries.values())
    
    def _evict_global_if_needed(self) -> None:
        """Evict LRU sensor when global entry limit exceeded (caller holds lock).
        
        Flushes data to callback before eviction if configured.
        """
        while self._get_total_entries() > self._max_total_entries and self._entries:
            evicted_id, entry = self._entries.popitem(last=False)
            
            # Flush before eviction to prevent data loss
            if self._flush_callback is not None:
                try:
                    readings = list(entry.window)
                    if readings:
                        self._flush_callback(evicted_id, readings)
                except Exception as e:
                    logger.warning(f"flush_failed sensor_id={evicted_id}: {e}")
            
            self._evictions_global += 1
            logger.debug("global_evict sensor_id=%d", evicted_id)
    
    def _cleanup_expired(self, now: float) -> None:
        """Remove entries idle longer than TTL (caller holds lock).
        
        Flushes data to callback before eviction if configured.
        """
        cutoff = now - self._ttl
        expired = [
            sid for sid, e in self._entries.items()
            if e.last_accessed < cutoff
        ]
        for sid in expired:
            entry = self._entries[sid]
            
            # Flush before eviction to prevent data loss
            if self._flush_callback is not None:
                try:
                    readings = list(entry.window)
                    if readings:
                        self._flush_callback(sid, readings)
                except Exception as e:
                    logger.warning(f"flush_failed sensor_id={sid}: {e}")
            
            del self._entries[sid]
            self._evictions_ttl += 1
            logger.debug("ttl_evict sensor_id=%d", sid)
    
    def _periodic_cleanup(self) -> None:
        """Background thread for proactive TTL cleanup."""
        interval = self._ttl / 2.0
        while not self._stop_event.wait(timeout=interval):
            now = time.monotonic()
            with self._lock:
                self._cleanup_expired(now)
    
    def close(self) -> None:
        """Stop cleanup thread and release resources."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_event.set()
            self._cleanup_thread.join(timeout=2.0)
            if self._cleanup_thread.is_alive():
                logger.warning("cleanup_thread_did_not_stop_gracefully")
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
    
    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
