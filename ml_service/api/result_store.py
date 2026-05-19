"""In-memory result store with TTL + LRU — FIX P3-3: async prediction results.

Environment variables:
  ML_RESULT_STORE_MAX_ENTRIES   — max entries (default: 10000)
  ML_RESULT_STORE_TTL_SECONDS   — TTL in seconds (default: 300)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = int(os.environ.get("ML_RESULT_STORE_MAX_ENTRIES", "10000"))
DEFAULT_TTL = int(os.environ.get("ML_RESULT_STORE_TTL_SECONDS", "300"))

# Sentinel: value not found or expired
_MISSING = object()


class PredictionResultStore:
    """LRU in-memory store for async prediction results with TTL."""

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = float(DEFAULT_TTL),
    ) -> None:
        self._max_entries = max(1, max_entries)
        self._ttl = float(ttl_seconds)
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def set(self, prediction_id: str, result: Any) -> None:
        """Store a result (success or error)."""
        with self._lock:
            self._entries[prediction_id] = {
                "result": result,
                "timestamp": time.monotonic(),
            }
            self._entries.move_to_end(prediction_id)
            self._evict_lru_if_needed()

    def get(self, prediction_id: str) -> Any:
        """Return result, _MISSING if not found/expired, or None if still pending."""
        with self._lock:
            entry = self._entries.get(prediction_id)
            if entry is None:
                self._misses += 1
                return _MISSING
            if time.monotonic() - entry["timestamp"] > self._ttl:
                del self._entries[prediction_id]
                self._misses += 1
                return _MISSING
            self._entries.move_to_end(prediction_id)
            self._hits += 1
            return entry["result"]

    def is_pending(self, prediction_id: str) -> bool:
        """True if the ID exists and result is None (still pending)."""
        with self._lock:
            entry = self._entries.get(prediction_id)
            if entry is None:
                return False
            if time.monotonic() - entry["timestamp"] > self._ttl:
                del self._entries[prediction_id]
                return False
            return entry.get("result") is None

    def _evict_lru_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get_metrics(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
            }

    @staticmethod
    def get_missing_sentinel() -> Any:
        """Expose _MISSING for route handlers to compare against."""
        return _MISSING
