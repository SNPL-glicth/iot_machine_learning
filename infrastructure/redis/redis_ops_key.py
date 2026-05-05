"""Redis Key Operations — key management operations."""

from __future__ import annotations

import logging
from typing import Any

try:
    from redis import Redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    Redis = Any
    RedisError = Exception
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisKeyOps:
    """Key management operations for Redis."""
    
    def __init__(self, redis_client: Redis):
        """Initialize key operations wrapper.
        
        Args:
            redis_client: Redis client instance
        """
        self._client = redis_client
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on key.
        
        Args:
            key: Redis key
            ttl: TTL in seconds
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._client.expire(key, ttl)
            return True
        except RedisError as e:
            logger.warning(f"Failed to set TTL on {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key.
        
        Args:
            key: Redis key
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            self._client.delete(key)
            return True
        except RedisError as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists.
        
        Args:
            key: Redis key
        
        Returns:
            True if exists, False otherwise
        """
        try:
            return self._client.exists(key) > 0
        except RedisError as e:
            logger.error(f"Failed to check existence {key}: {e}")
            return False
