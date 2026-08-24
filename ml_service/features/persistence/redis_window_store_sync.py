"""Sync Redis window store operations."""

from __future__ import annotations

import json
import logging
import time
from typing import List, Optional, Any

from iot_machine_learning.infrastructure.persistence.redis import (
    RedisConnectionManager,
    get_redis_circuit_breaker,
)

from .redis_window_models import PersistedWindow

logger = logging.getLogger(__name__)


class SyncRedisWindowStore:
    """Sync Redis operations for window persistence."""
    
    KEY_PREFIX = "zenin:window:sensor:"
    DEFAULT_TTL = 3600
    
    def __init__(
        self,
        redis_client: Any,
        ttl_seconds: int = DEFAULT_TTL,
        max_age_seconds: int = 120,
    ):
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._max_age = max_age_seconds
        self._circuit_breaker = get_redis_circuit_breaker("redis_window")
    
    def _key(self, sensor_id: int) -> str:
        return f"{self.KEY_PREFIX}{sensor_id}"
    
    def _save(self, sensor_id: int, values: List[float], timestamps: List[float]) -> bool:
        window = PersistedWindow(
            sensor_id=sensor_id,
            values=values[-100:],
            timestamps=timestamps[-100:],
            last_updated=time.time(),
        )
        data = json.dumps(window.__dict__)
        pipe = self._redis.pipeline()
        pipe.set(self._key(sensor_id), data)
        pipe.expire(self._key(sensor_id), self._ttl)
        pipe.execute()
        return True
    
    def save(self, sensor_id: int, values: List[float], timestamps: List[float]) -> bool:
        def _fallback():
            logger.debug("Window save skipped (circuit open): sensor_id=%d", sensor_id)
            return False
        try:
            return self._circuit_breaker.call(
                lambda: self._save(sensor_id, values, timestamps), _fallback
            )
        except Exception as e:
            logger.warning("Failed to save window: sensor_id=%d error=%s", sensor_id, e)
            return False
    
    def _load(self, sensor_id: int) -> Optional[PersistedWindow]:
        data = self._redis.get(self._key(sensor_id))
        if data is None:
            return None
        parsed = json.loads(data)
        window = PersistedWindow(**parsed)
        age = time.time() - window.last_updated
        if age > self._max_age:
            logger.warning("Stale window rejected: sensor_id=%d age=%.1fs", sensor_id, age)
            self._redis.delete(self._key(sensor_id))
            return None
        return window
    
    def load(self, sensor_id: int, max_age_seconds: Optional[int] = None) -> Optional[PersistedWindow]:
        original_max_age = self._max_age
        if max_age_seconds is not None:
            self._max_age = max_age_seconds
        def _fallback():
            logger.debug("Window load skipped (circuit open): sensor_id=%d", sensor_id)
            return None
        try:
            result = self._circuit_breaker.call(lambda: self._load(sensor_id), _fallback)
            return result
        except Exception as e:
            logger.warning("Failed to load window: sensor_id=%d error=%s", sensor_id, e)
            return None
        finally:
            self._max_age = original_max_age
    
    def delete(self, sensor_id: int) -> bool:
        try:
            self._redis.delete(self._key(sensor_id))
            return True
        except Exception:
            return False
    
    def get_all_sensor_ids(self) -> List[int]:
        pattern = f"{self.KEY_PREFIX}*"
        sensor_ids = []
        prefix_len = len(self.KEY_PREFIX)
        for key in self._redis.scan_iter(match=pattern, count=100):
            key_str = key.decode() if isinstance(key, bytes) else key
            sensor_id_str = key_str[prefix_len:]
            try:
                sensor_ids.append(int(sensor_id_str))
            except ValueError:
                pass
        return sensor_ids