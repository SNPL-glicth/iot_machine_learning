"""Redis window store - main entry point with singleton."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Any

from .redis_window_models import PersistedWindow
from .redis_window_store_sync import SyncRedisWindowStore
from .redis_window_store_async import AsyncRedisWindowStore

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisWindowStore:
    """Main Redis window store combining sync and async operations."""
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        ttl_seconds: int = 3600,
        max_age_seconds: int = 120,
    ):
        self._sync = SyncRedisWindowStore(redis_client, ttl_seconds, max_age_seconds)
        self._async = AsyncRedisWindowStore(redis_client, ttl_seconds, max_age_seconds)
        self._enabled = redis_client is not None
        
        if self._enabled:
            logger.info("ML WindowStore initialized (ttl=%ds, max_age=%ds)", ttl_seconds, max_age_seconds)
        else:
            logger.warning("ML WindowStore disabled (no Redis client)")
    
    # Delegate to sync
    def save(self, sensor_id: int, values: List[float], timestamps: List[float]) -> bool:
        return self._sync.save(sensor_id, values, timestamps)
    
    def load(self, sensor_id: int, max_age_seconds: Optional[int] = None) -> Optional[PersistedWindow]:
        return self._sync.load(sensor_id, max_age_seconds)
    
    def delete(self, sensor_id: int) -> bool:
        return self._sync.delete(sensor_id)
    
    def get_all_sensor_ids(self) -> List[int]:
        return self._sync.get_all_sensor_ids()
    
    # Delegate to async
    async def save_async(self, sensor_id: int, values: List[float], timestamps: List[float]) -> bool:
        return await self._async.save(sensor_id, values, timestamps)
    
    async def load_async(self, sensor_id: int, max_age_seconds: Optional[int] = None) -> Optional[PersistedWindow]:
        return await self._async.load(sensor_id, max_age_seconds)
    
    async def delete_async(self, sensor_id: int) -> bool:
        return await self._async.delete(sensor_id)
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    @property
    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "ttl_seconds": self._sync._ttl,
            "max_age_seconds": self._sync._max_age,
            "sensor_count": len(self._sync.get_all_sensor_ids()) if self._enabled else 0,
            "circuit_breaker": self._sync._circuit_breaker.get_metrics(),
        }


# Singleton
_store_instance: Optional[RedisWindowStore] = None


def get_window_store() -> RedisWindowStore:
    """Get singleton window store."""
    global _store_instance
    
    if _store_instance is not None:
        return _store_instance
    
    if not REDIS_AVAILABLE:
        logger.warning("ML WindowStore: redis package not installed")
        _store_instance = RedisWindowStore(redis_client=None)
        return _store_instance
    
    ttl = int(os.getenv("ML_WINDOW_TTL_SECONDS", "3600"))
    max_age = int(os.getenv("ML_WINDOW_MAX_AGE_SECONDS", "120"))
    
    try:
        from iot_machine_learning.infrastructure.persistence.redis import RedisConnectionManager
        client = RedisConnectionManager.get_sync_client()
        logger.info("ML WindowStore connected (ttl=%ds, max_age=%ds)", ttl, max_age)
        _store_instance = RedisWindowStore(client, ttl_seconds=ttl, max_age_seconds=max_age)
    except Exception as e:
        logger.warning("ML WindowStore connection failed: %s", e)
        _store_instance = RedisWindowStore(redis_client=None)
    
    return _store_instance


def reset_window_store() -> None:
    """Reset singleton for tests."""
    global _store_instance
    _store_instance = None


__all__ = ["RedisWindowStore", "PersistedWindow", "get_window_store", "reset_window_store"]