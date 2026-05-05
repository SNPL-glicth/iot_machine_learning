"""Redis Collection Operations — hash, list, and sorted set operations."""

from __future__ import annotations

import logging
from typing import Any, List

try:
    from redis import Redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    Redis = Any
    RedisError = Exception
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisHashOps:
    """Hash operations for Redis."""
    
    def __init__(self, redis_client: Redis):
        self._client = redis_client
    
    def hset(self, key: str, field: str, value: str) -> bool:
        """Set hash field."""
        try:
            self._client.hset(key, field, value)
            return True
        except RedisError as e:
            logger.error(f"Failed to hset {key}:{field}: {e}")
            return False
    
    def hget(self, key: str, field: str) -> str | None:
        """Get hash field value."""
        try:
            value = self._client.hget(key, field)
            return value.decode('utf-8') if value else None
        except RedisError as e:
            logger.error(f"Failed to hget {key}:{field}: {e}")
            return None
    
    def hgetall(self, key: str) -> dict:
        """Get all hash fields."""
        try:
            data = self._client.hgetall(key)
            return {k.decode('utf-8'): v.decode('utf-8') for k, v in data.items()}
        except RedisError as e:
            logger.error(f"Failed to hgetall {key}: {e}")
            return {}


class RedisListOps:
    """List operations for Redis."""
    
    def __init__(self, redis_client: Redis):
        self._client = redis_client
    
    def lpush(self, key: str, value: str) -> bool:
        """Push to list head."""
        try:
            self._client.lpush(key, value)
            return True
        except RedisError as e:
            logger.error(f"Failed to lpush {key}: {e}")
            return False
    
    def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim list to range."""
        try:
            self._client.ltrim(key, start, end)
            return True
        except RedisError as e:
            logger.error(f"Failed to ltrim {key}: {e}")
            return False
    
    def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get list range."""
        try:
            values = self._client.lrange(key, start, end)
            return [v.decode('utf-8') for v in values]
        except RedisError as e:
            logger.error(f"Failed to lrange {key}: {e}")
            return []


class RedisSortedSetOps:
    """Sorted set operations for Redis."""
    
    def __init__(self, redis_client: Redis):
        self._client = redis_client
    
    def zadd(self, key: str, score: float, member: str) -> bool:
        """Add to sorted set."""
        try:
            self._client.zadd(key, {member: score})
            return True
        except RedisError as e:
            logger.error(f"Failed to zadd {key}: {e}")
            return False
    
    def zrange(self, key: str, start: int = 0, end: int = -1, withscores: bool = False) -> List:
        """Get sorted set range."""
        try:
            values = self._client.zrange(key, start, end, withscores=withscores)
            if withscores:
                return [(v[0].decode('utf-8'), v[1]) for v in values]
            return [v.decode('utf-8') for v in values]
        except RedisError as e:
            logger.error(f"Failed to zrange {key}: {e}")
            return []
