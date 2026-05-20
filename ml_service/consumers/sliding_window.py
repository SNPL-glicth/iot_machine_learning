"""In-memory sliding window store per series_id with LRU + TTL eviction.
FIX P3-2: Sharded locks. Environment: ML_WINDOW_STORE_SHARDS (default: 16)."""
from __future__ import annotations
import atexit
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from iot_machine_learning.domain.entities.iot.sensor_reading import Reading

logger = logging.getLogger(__name__)
DEFAULT_SHARDS = int(os.environ.get("ML_WINDOW_STORE_SHARDS", "16"))

@dataclass
class _WindowEntry:
    window: deque
    last_accessed: float = field(default_factory=time.monotonic)

class _Shard:
    __slots__ = ("entries", "lock", "evictions_lru", "evictions_ttl", "evictions_global")
    def __init__(self):
        self.entries: OrderedDict[str, _WindowEntry] = OrderedDict()
        self.lock = threading.Lock()
        self.evictions_lru = 0
        self.evictions_ttl = 0
        self.evictions_global = 0

class SlidingWindowStore:
    """LRU + TTL cache with sharded locks (P3-2).

    FIX P3-6: Si distributed_adapter está presente, append escribe en
    AMBOS (local + Redis) para que otras réplicas vean la ventana.
    """
    def __init__(
        self,
        max_size: int = 20,
        max_sensors: int = 1000,
        ttl_seconds: float = 3600.0,
        enable_proactive_cleanup: bool = True,
        max_total_entries: int = 50000,
        flush_callback: Optional[Callable[[str, List[Reading]], None]] = None,
        n_shards: int = DEFAULT_SHARDS,
        distributed_adapter=None,
    ) -> None:
        self._max_size = max_size
        self._max_sensors = max(1, max_sensors)
        self._max_total_entries = max(1, max_total_entries)
        self._ttl = float(ttl_seconds)
        self._n_shards = max(1, n_shards)
        self._shards = [_Shard() for _ in range(self._n_shards)]
        self._flush_callback = flush_callback
        self._distributed = distributed_adapter
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        logger.info("[SLIDING_WINDOW] n_shards=%d distributed=%s",
                    self._n_shards, bool(self._distributed))
        if enable_proactive_cleanup and ttl_seconds > 0:
            self._cleanup_thread = threading.Thread(
                target=self._periodic_cleanup, daemon=True, name="SlidingWindowCleanup",
            )
            self._cleanup_thread.start()
            atexit.register(self.close)
    def _shard_idx(self, series_id: str) -> int:
        return hash(str(series_id)) % self._n_shards
    def _get_shard(self, series_id: str) -> _Shard:
        return self._shards[self._shard_idx(series_id)]
    def append(self, reading: Reading) -> int:
        sid = str(reading.series_id)
        now = time.monotonic()
        shard = self._get_shard(sid)
        with shard.lock:
            self._cleanup_expired(shard, now)
            if sid in shard.entries:
                entry = shard.entries[sid]
                shard.entries.move_to_end(sid)
            else:
                entry = _WindowEntry(window=deque(maxlen=self._max_size), last_accessed=now)
                shard.entries[sid] = entry
                self._evict_lru_if_needed(shard)
            entry.window.append(reading)
            entry.last_accessed = now
            self._evict_global_if_needed(shard)
            count = len(entry.window)
        # FIX P3-6: replica a Redis para otras réplicas
        if self._distributed is not None:
            try:
                self._distributed.append(reading)
            except Exception as e:
                logger.error("[P3-6] distributed_append_failed sensor=%s: %s", sid, e)
        return count
    def get_window(self, series_id: str) -> List[Reading]:
        sid = str(series_id)
        shard = self._get_shard(sid)
        with shard.lock:
            entry = shard.entries.get(sid)
            if entry is None:
                return []
            shard.entries.move_to_end(sid)
            entry.last_accessed = time.monotonic()
            readings = list(entry.window)
        readings.sort(key=lambda r: r.timestamp)
        return readings
    def clear(self, series_id: str) -> None:
        sid = str(series_id)
        shard = self._get_shard(sid)
        with shard.lock:
            entry = shard.entries.get(sid)
            if entry is not None:
                entry.window.clear()
    @property
    def series_ids(self) -> List[str]:
        result = []
        for shard in self._shards:
            with shard.lock:
                result.extend(shard.entries.keys())
        return result
    def get_metrics(self) -> Dict:
        total = 0; lru = ttl = glob = 0; active = 0
        for shard in self._shards:
            with shard.lock:
                active += len(shard.entries)
                total += sum(len(e.window) for e in shard.entries.values())
                lru += shard.evictions_lru; ttl += shard.evictions_ttl; glob += shard.evictions_global
        return {"active_sensors": active, "max_sensors": self._max_sensors, "evictions_lru": lru,
                "evictions_ttl": ttl, "evictions_global": glob, "total_entries": total, "n_shards": self._n_shards}

    def _flush(self, sid: str, entry: _WindowEntry) -> None:
        if self._flush_callback is not None and entry.window:
            try:
                self._flush_callback(sid, list(entry.window))
            except Exception as e:
                logger.warning("flush_failed series_id=%s: %s", sid, e)
    def _evict_lru_if_needed(self, shard: _Shard) -> None:
        per_shard = max(1, (self._max_sensors + self._n_shards - 1) // self._n_shards)
        while len(shard.entries) > per_shard and shard.entries:
            evicted_id, entry = shard.entries.popitem(last=False)
            self._flush(evicted_id, entry)
            shard.evictions_lru += 1
    def _evict_global_if_needed(self, shard: _Shard) -> None:
        total = sum(len(e.window) for e in shard.entries.values())
        if total <= self._max_total_entries // self._n_shards:
            return
        true_total = 0
        for s in self._shards:
            with s.lock:
                true_total += sum(len(e.window) for e in s.entries.values())
        while true_total > self._max_total_entries and shard.entries:
            evicted_id, entry = shard.entries.popitem(last=False)
            self._flush(evicted_id, entry)
            shard.evictions_global += 1
            logger.debug("global_evict series_id=%s", evicted_id)
            true_total -= len(entry.window)
    def _cleanup_expired(self, shard: _Shard, now: float) -> None:
        cutoff = now - self._ttl
        expired = [sid for sid, e in shard.entries.items() if e.last_accessed < cutoff]
        for sid in expired:
            entry = shard.entries[sid]
            self._flush(sid, entry)
            del shard.entries[sid]
            shard.evictions_ttl += 1
    def _periodic_cleanup(self) -> None:
        interval = self._ttl / 2.0
        while not self._stop_event.wait(timeout=interval):
            now = time.monotonic()
            for shard in self._shards:
                with shard.lock:
                    self._cleanup_expired(shard, now)
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
