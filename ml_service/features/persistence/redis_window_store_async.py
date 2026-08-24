"""Async Redis window store operations."""

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


class AsyncRedisWindowStore:
    """Async Redis operations for window persistence."""
    
    KEY_PREFIX = "zenin:window:sensor:"
    DEFAULT_TTL = 3600
    
    def __init__(
        self,
        redis_client: Any = None,
        ttl_seconds: int = DEFAULT_TTL,
        max_age_seconds: int = 120,
    ):
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._max_age = max_age_seconds
        self._circuit_breaker = get_redis_circuit_breaker("redis_window")
    
    def _key(self, sensor_id: int) -> str:
        return f"{self.KEY_PREFIX}{sensor_id}"
    
    async def save(
        self,
        sensor_id: int,
        values: List[float],
        timestamps: List[float]
    ) -> bool:
        if not self._redis:
            return False
        
        def _fallback():
            logger.debug("Window save skipped (circuit open): sensor_id=%d", sensor_id)
            return False
        
        try:
            window = PersistedWindow(
                sensor_id=sensor_id,
                values=values[-100:],
                timestamps=timestamps[-100:],
                last_updated=time.time(),
            )
            data = json.dumps(window.__dict__)
            async_client = await RedisConnectionManager.get_async_client()
            await async_client.setex(self._key(sensor_id), self._ttl, data.encode())
            return True
        except Exception as e:
            logger.warning("Failed to save window (async): sensor_id=%d error=%s", sensor_id, e)
            return False
    
    async def load(
        self,
        sensor_id: int,
        max_age_seconds: Optional[int] = None
    ) -> Optional[PersistedWindow]:
        if not self._redis:
            return None
        
        max_age = max_age_seconds or self._max_age
        
        try:
            async_client = await RedisConnectionManager.get_async_client()
            data = await async_client.get(self._key(sensor_id))
            if data is None:
                return None
            
            parsed = json.loads(data.decode())
            window = PersistedWindow(**parsed)
            
            age = time.time() - window.last_updated
            if age > max_age:
                logger.warning("Stale window rejected (async): sensor_id=%d age=%.1fs", sensor_id, age)
                await async_client.delete(self._key(sensor_id))
                return None
            
            return window
            
        except Exception as e:
            logger.warning("Failed to load window (async): sensor_id=%d error=%s", sensor_id, e)
            return None
    
    async def delete(self, sensor_id: int) -> bool:
        try:
            async_client = await RedisConnectionManager.get_async_client()
            await async_client.delete(self._key(sensor_id))
            return True
        except Exception:
            return False
    
    async def get_all_sensor_ids(self) -> List[int]:
        return []  # Not implemented for async