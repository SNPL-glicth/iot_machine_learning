"""EngineErrorStore — single source of truth for engine prediction errors.

Class invariant
---------------
A raw prediction error float (``abs(predicted - actual)``) is written to
Redis **exactly once**, through :py:meth:`EngineErrorStore.record`.
Readers consume through :py:meth:`get_recent`, :py:meth:`get_percentile`,
or :py:meth:`get_rmse_window`; no reader issues direct Redis commands
against the ``error_store:*`` keyspace.

Persistence contract
--------------------
* Key pattern:  ``{prefix}:{series_id}:{engine_name}``   (prefix default = ``error_store``)
* Type:         Redis List (``RPUSH`` / ``LTRIM`` / ``LRANGE``)
* Element:      a decimal string repr of a non-negative float
* Cap:          last ``max_entries`` entries (default 200)
* TTL:          ``ttl_seconds`` (default 30 days), reset on every write

Failure mode
------------
If no Redis client is provided, or a Redis call raises, the store
silently falls back to a thread-safe bounded in-memory deque per
``(series_id, engine_name)`` pair. The public API never raises.

SRP
---
Stores, trims, and exposes recent errors. Does not classify engines,
emit alerts, or mutate any engine state.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_DEFAULT_MAX_ENTRIES: int = 200
_DEFAULT_TTL_SECONDS: int = 30 * 24 * 3600  # 30 days
_DEFAULT_PREFIX: str = "error_store"
_DEFAULT_MAX_SERIES: int = 10_000


class EngineErrorStore:
    """Redis-backed error store with in-memory fallback.

    Args:
        redis_client: Optional Redis client; when ``None`` the store is
            purely in-memory (useful for tests and degraded environments).
        key_prefix: Redis key namespace. Injectable for test isolation.
        max_entries: Hard cap on stored entries per (series, engine).
        ttl_seconds: TTL reset on every successful write.
        max_series_engines: Upper bound on tracked (series, engine) pairs
            in the in-memory fallback; LRU eviction beyond this.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        *,
        key_prefix: str = _DEFAULT_PREFIX,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_series_engines: int = _DEFAULT_MAX_SERIES,
    ) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")

        self._redis = redis_client
        self._prefix = key_prefix
        self._max_entries = int(max_entries)
        self._ttl = int(ttl_seconds)
        self._max_pairs = int(max_series_engines)

        self._memory: "OrderedDict[Tuple[str, str], Deque[float]]" = OrderedDict()
        self._lock = threading.RLock()

    # -- public API ---------------------------------------------------

    def record(self, series_id: str, engine_name: str, error: float) -> None:
        """Persist one error. Never raises.

        ``error`` must be a finite, non-negative number; NaN/Inf/negative
        values are silently dropped with a debug log (defensive — upstream
        should already have sanitized).
        """
        if not self._is_valid_error(error):
            logger.debug(
                "error_store_invalid_value_dropped",
                extra={"series_id": series_id, "engine": engine_name, "error": error},
            )
            return

        e = float(error)
        if self._write_redis(series_id, engine_name, e):
            return
        self._write_memory(series_id, engine_name, e)

    def get_recent(self, series_id: str, engine_name: str, n: int) -> List[float]:
        """Return the last ``n`` errors in chronological order (oldest first)."""
        if n <= 0:
            return []
        values = self._read(series_id, engine_name)
        return values[-n:] if len(values) > n else values

    def get_percentile(self, series_id: str, engine_name: str, p: float) -> float:
        """Return the ``p``-th percentile (0-100) over all stored errors.

        Uses linear interpolation. Returns ``0.0`` when there is no data —
        consumers must interpret ``0.0`` as "insufficient history".
        """
        if not 0.0 <= p <= 100.0:
            raise ValueError(f"p must be in [0, 100], got {p}")
        values = self._read(series_id, engine_name)
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        k = (p / 100.0) * (len(ordered) - 1)
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return float(ordered[lo])
        frac = k - lo
        return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)

    def get_rmse_window(self, series_id: str, engine_name: str, window: int) -> float:
        """Root-mean-square of the last ``window`` stored errors.

        Returns ``0.0`` when no data is available.
        """
        values = self.get_recent(series_id, engine_name, window)
        if not values:
            return 0.0
        sq = sum(v * v for v in values)
        return math.sqrt(sq / len(values))

    # -- helpers ------------------------------------------------------

    def _key(self, series_id: str, engine_name: str) -> str:
        return f"{self._prefix}:{series_id}:{engine_name}"

    @staticmethod
    def _is_valid_error(error: float) -> bool:
        try:
            f = float(error)
        except (TypeError, ValueError):
            return False
        return math.isfinite(f) and f >= 0.0

    def _write_redis(self, series_id: str, engine_name: str, e: float) -> bool:
        if self._redis is None:
            return False
        try:
            key = self._key(series_id, engine_name)
            pipe = self._redis.pipeline()
            pipe.rpush(key, e)
            pipe.ltrim(key, -self._max_entries, -1)
            pipe.expire(key, self._ttl)
            pipe.execute()
            return True
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "error_store_redis_write_failed",
                extra={"series_id": series_id, "engine": engine_name, "error": str(exc)},
            )
            return False

    def _write_memory(self, series_id: str, engine_name: str, e: float) -> None:
        with self._lock:
            pair = (series_id, engine_name)
            buf = self._memory.get(pair)
            if buf is None:
                if len(self._memory) >= self._max_pairs:
                    self._memory.popitem(last=False)  # LRU evict oldest
                buf = deque(maxlen=self._max_entries)
                self._memory[pair] = buf
            else:
                self._memory.move_to_end(pair)
            buf.append(e)

    def _read(self, series_id: str, engine_name: str) -> List[float]:
        if self._redis is not None:
            try:
                raw = self._redis.lrange(self._key(series_id, engine_name), 0, -1)
                if raw is not None:
                    return [float(x) for x in raw]
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "error_store_redis_read_failed",
                    extra={"series_id": series_id, "engine": engine_name, "error": str(exc)},
                )
        with self._lock:
            buf = self._memory.get((series_id, engine_name))
            return list(buf) if buf else []
