"""In-memory sliding window store per series_id with LRU + TTL eviction."""
from __future__ import annotations
import atexit
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
logger = logging.getLogger(__name__)
@dataclass(frozen=True)
class SlidingWindowConfig:
    max_sensors: int = 1000
    ttl_seconds: float = 3600.0
    cleanup_enabled: bool = True
@dataclass
class _WindowEntry:
    window: deque
    last_accessed: float = field(default_factory=time.monotonic)
class SlidingWindowStore:
    """LRU + TTL cache of sliding windows per series_id."""
    def __init__(
        self,
        max_size: int = 20,
        max_sensors: int = 1000,
        ttl_seconds: float = 3600.0,
        enable_proactive_cleanup: bool = True,
        max_total_entries: int = 50000,
        flush_callback: Optional[Callable[[str, List[Reading]], None]] = None,
    ) -> None:
        self._max_size = max_size
        self._max_sensors = max(1, max_sensors)
        self._max_total_entries = max(1, max_total_entries)
        self._ttl = float(ttl_seconds)
        self._entries: OrderedDict[str, _WindowEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._evictions_lru: int = 0
        self._evictions_ttl: int = 0
        self._evictions_global: int = 0
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._flush_callback = flush_callback
        if enable_proactive_cleanup and ttl_seconds > 0:
            self._cleanup_thread = threading.Thread(
                target=self._periodic_cleanup,
                daemon=True,
                name="SlidingWindowCleanup"
            )
            self._cleanup_thread.start()
            atexit.register(self.close)
    def append(self, reading: Reading) -> int:
        sid = reading.series_id
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
    def get_window(self, series_id: str) -> List[Reading]:
        with self._lock:
            entry = self._entries.get(series_id)
            if entry is None:
                return []
            self._entries.move_to_end(series_id)
            entry.last_accessed = time.monotonic()
            readings = list(entry.window)
        readings.sort(key=lambda r: r.timestamp)
        return readings
    def clear(self, series_id: str) -> None:
        with self._lock:
            entry = self._entries.get(series_id)
            if entry is not None:
                entry.window.clear()
    @property
    def series_ids(self) -> List[str]:
        with self._lock:
            return list(self._entries.keys())
    def get_metrics(self) -> Dict:
        with self._lock:
            return {
                "active_sensors": len(self._entries),
                "max_sensors": self._max_sensors,
                "evictions_lru": self._evictions_lru,
                "evictions_ttl": self._evictions_ttl,
                "evictions_global": self._evictions_global,
                "total_entries": self._get_total_entries(),
            }
    def _evict_lru_if_full(self) -> None:
        while len(self._entries) >= self._max_sensors:
            evicted_id, entry = self._entries.popitem(last=False)
            if self._flush_callback is not None:
                try:
                    readings = list(entry.window)
                    if readings:
                        self._flush_callback(evicted_id, readings)
                except Exception as e:
                    logger.warning(f"flush_failed series_id={evicted_id}: {e}")
            self._evictions_lru += 1
            logger.debug("lru_evict series_id=%s", evicted_id)
    def _evict_global_if_needed(self) -> None:
        """Evict LRU sensor when global entry limit exceeded (caller holds lock)."""
        while self._get_total_entries() > self._max_total_entries and self._entries:
            evicted_id, entry = self._entries.popitem(last=False)
            if self._flush_callback is not None:
                try:
                    readings = list(entry.window)
                    if readings:
                        self._flush_callback(evicted_id, readings)
                except Exception as e:
                    logger.warning(f"flush_failed series_id={evicted_id}: {e}")
            self._evictions_global += 1
            logger.debug("global_evict series_id=%s", evicted_id)
    def _cleanup_expired(self, now: float) -> None:
        cutoff = now - self._ttl
        expired = [
            sid for sid, e in self._entries.items()
            if e.last_accessed < cutoff
        ]
        for sid in expired:
            entry = self._entries[sid]
            if self._flush_callback is not None:
                try:
                    readings = list(entry.window)
                    if readings:
                        self._flush_callback(sid, readings)
                except Exception as e:
                    logger.warning(f"flush_failed series_id={sid}: {e}")
            del self._entries[sid]
            self._evictions_ttl += 1
            logger.debug("ttl_evict series_id=%s", sid)
    def _periodic_cleanup(self) -> None:
        interval = self._ttl / 2.0
        while not self._stop_event.wait(timeout=interval):
            now = time.monotonic()
            with self._lock:
                self._cleanup_expired(now)
    def close(self) -> None:
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
        try:
            self.close()
        except Exception:
            pass
