"""FIX P3-6: Ventana deslizante distribuida sobre Redis Sorted Sets.

Variables de entorno:
  ML_DISTRIBUTED_WINDOWS_ENABLED   (default: false)
  ML_DIST_WINDOW_TTL_SECONDS       (default: 3600)
  ML_DIST_WINDOW_MAX_ENTRIES       (default: 1000)
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

from .clients import get_sync_redis
from .circuit_factory import get_redis_circuit_breaker

logger = logging.getLogger(__name__)

KEY_FMT = "dist_window:{sensor_id}"


def _is_enabled() -> bool:
    return os.environ.get("ML_DISTRIBUTED_WINDOWS_ENABLED", "false").lower() in ("1", "true", "yes")


def _ttl() -> int:
    return int(os.environ.get("ML_DIST_WINDOW_TTL_SECONDS", "3600"))


def _max_entries() -> int:
    return int(os.environ.get("ML_DIST_WINDOW_MAX_ENTRIES", "1000"))


class DistributedWindowAdapter:
    """Ventana deslizante distribuida. FIX PROD-1: circuit breaker."""

    def __init__(self, redis=None) -> None:
        self._redis = redis
        self._enabled = _is_enabled()
        self._ttl = _ttl()
        self._max = _max_entries()
        if self._enabled and self._redis is None:
            try:
                self._redis = get_sync_redis()
            except Exception as e:
                logger.warning("[P3-6] DistributedWindow init failed: %s", e)
                self._enabled = False
        self._circuit = get_redis_circuit_breaker("dist_window_adapter")
        self._last_cb_state = self._circuit.state.value
        if self._enabled:
            logger.info("[P3-6] DistributedWindow enabled ttl=%ds max=%d", self._ttl, self._max)

    def _log_transition(self) -> None:
        current = self._circuit.state.value
        if current == self._last_cb_state:
            return
        if self._last_cb_state == "closed" and current == "open":
            m = self._circuit.get_metrics()
            logger.critical("[PROD-1] dist_window_circuit_opened failure_count=%d", m["failure_count"])
        elif self._last_cb_state == "open" and current == "half_open":
            logger.warning("[PROD-1] dist_window_circuit_probing")
        elif self._last_cb_state == "half_open" and current == "closed":
            logger.info("[PROD-1] dist_window_circuit_recovered")
        self._last_cb_state = current

    def append(self, sensor_id: int, value: float, timestamp: Optional[float] = None) -> None:
        if not self._enabled or self._redis is None:
            return
        self._circuit.call(
            lambda: self._append_impl(sensor_id, value, timestamp),
            fallback=lambda: None,
        )
        self._log_transition()

    def _append_impl(self, sensor_id: int, value: float, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        key = KEY_FMT.format(sensor_id=sensor_id)
        pipe = self._redis.pipeline()
        pipe.zadd(key, {value: ts})
        pipe.zremrangebyrank(key, 0, -(self._max + 1))
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get(self, sensor_id: int, n: int) -> List[Tuple[float, float]]:
        if not self._enabled or self._redis is None:
            return []
        result = self._circuit.call(
            lambda: self._get_impl(sensor_id, n),
            fallback=lambda: [],
        )
        self._log_transition()
        return result

    def _get_impl(self, sensor_id: int, n: int) -> List[Tuple[float, float]]:
        key = KEY_FMT.format(sensor_id=sensor_id)
        rows = self._redis.zrevrange(key, 0, n - 1, withscores=True)
        return [(float(ts), float(v)) for v, ts in rows]

    def remove(self, sensor_id: int, count: int) -> None:
        if not self._enabled or self._redis is None:
            return
        self._circuit.call(
            lambda: self._redis.zremrangebyrank(KEY_FMT.format(sensor_id=sensor_id), 0, count - 1),
            fallback=lambda: None,
        )
        self._log_transition()

    def __len__(self) -> int:
        if not self._enabled or self._redis is None:
            return 0
        result = self._circuit.call(
            lambda: len(self._redis.keys("dist_window:*") or []),
            fallback=lambda: 0,
        )
        self._log_transition()
        return result
