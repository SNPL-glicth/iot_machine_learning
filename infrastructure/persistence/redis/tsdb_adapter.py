"""FIX P3-5: Redis TSDB adapter — liviano con Sorted Sets.

Variables de entorno:
  ML_TSDB_ENABLED              (default: false)
  ML_TSDB_TTL_SECONDS          (default: 3600)
  ML_TSDB_MAX_ENTRIES          (default: 1000)
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

from .clients import get_sync_redis
from .circuit_factory import get_redis_circuit_breaker
from .circuit_breaker import CircuitState

logger = logging.getLogger(__name__)

KEY_FMT = "tsdb:readings:{sensor_id}"


def _is_enabled() -> bool:
    return os.environ.get("ML_TSDB_ENABLED", "false").lower() in ("1", "true", "yes")


def _ttl() -> int:
    return int(os.environ.get("ML_TSDB_TTL_SECONDS", "3600"))


def _max_entries() -> int:
    return int(os.environ.get("ML_TSDB_MAX_ENTRIES", "1000"))


def _failure_threshold() -> int:
    return int(os.environ.get("ML_TSDB_CIRCUIT_FAILURE_THRESHOLD", "5"))


def _recovery_timeout() -> int:
    return int(os.environ.get("ML_TSDB_CIRCUIT_RECOVERY_TIMEOUT_SECONDS", "30"))


class RedisTSDBAdapter:
    """Liviano TSDB sobre Redis Sorted Sets. FIX PROD-1: circuit breaker."""

    def __init__(self, redis=None) -> None:
        self._redis = redis
        self._enabled = _is_enabled()
        self._ttl = _ttl()
        self._max = _max_entries()
        if self._enabled and self._redis is None:
            try:
                self._redis = get_sync_redis()
            except Exception as e:
                logger.warning("[P3-5] TSDB init failed: %s", e)
                self._enabled = False
        self._circuit = get_redis_circuit_breaker(
            "tsdb_adapter",
            failure_threshold=_failure_threshold(),
            recovery_timeout=_recovery_timeout(),
        )
        self._last_cb_state = self._circuit.state.value
        if self._enabled:
            logger.info("[P3-5] TSDB enabled ttl=%ds max=%d", self._ttl, self._max)

    def _log_transition(self) -> None:
        current = self._circuit.state.value
        if current == self._last_cb_state:
            return
        if self._last_cb_state == "closed" and current == "open":
            m = self._circuit.get_metrics()
            logger.critical("[PROD-1] tsdb_circuit_opened failure_count=%d", m["failure_count"])
        elif self._last_cb_state == "open" and current == "half_open":
            logger.warning("[PROD-1] tsdb_circuit_probing")
        elif self._last_cb_state == "half_open" and current == "closed":
            logger.info("[PROD-1] tsdb_circuit_recovered")
        self._last_cb_state = current

    def append(self, sensor_id: int, value: float, timestamp: float) -> None:
        if not self._enabled or self._redis is None:
            return
        self._circuit.call(
            lambda: self._append_impl(sensor_id, value, timestamp),
            fallback=lambda: None,
        )
        self._log_transition()

    def _append_impl(self, sensor_id: int, value: float, timestamp: float) -> None:
        key = KEY_FMT.format(sensor_id=sensor_id)
        payload = json.dumps({"v": value, "ts": timestamp})
        pipe = self._redis.pipeline()
        pipe.zadd(key, {payload: timestamp})
        pipe.zremrangebyrank(key, 0, -(self._max + 1))
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get_recent(self, sensor_id: int, n: int) -> List[Tuple[float, float]]:
        if not self._enabled or self._redis is None:
            return []
        result = self._circuit.call(
            lambda: self._get_recent_impl(sensor_id, n),
            fallback=lambda: [],
        )
        self._log_transition()
        return result

    def _get_recent_impl(self, sensor_id: int, n: int) -> List[Tuple[float, float]]:
        key = KEY_FMT.format(sensor_id=sensor_id)
        rows = self._redis.zrevrange(key, 0, n - 1, withscores=True)
        result = []
        for payload, ts in rows:
            if isinstance(payload, bytes):
                payload = payload.decode()
            data = json.loads(payload)
            result.append((float(ts), float(data["v"])))
        return result

    def count(self, sensor_id: int) -> int:
        if not self._enabled or self._redis is None:
            return 0
        result = self._circuit.call(
            lambda: self._redis.zcard(KEY_FMT.format(sensor_id=sensor_id)),
            fallback=lambda: 0,
        )
        self._log_transition()
        return result

    def flush_sensor(self, sensor_id: int) -> None:
        if not self._enabled or self._redis is None:
            return
        self._circuit.call(
            lambda: self._redis.delete(KEY_FMT.format(sensor_id=sensor_id)),
            fallback=lambda: None,
        )
        self._log_transition()
