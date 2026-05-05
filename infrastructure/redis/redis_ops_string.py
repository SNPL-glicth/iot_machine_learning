"""Redis String Operations — string and counter operations with error handling."""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from redis import Redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    Redis = Any
    RedisError = Exception
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisStringOps:
    """String and counter operations for Redis."""
    
    def __init__(self, redis_client: Redis):
        """Initialize string operations wrapper.
        
        Args:
            redis_client: Redis client instance
        """
        self._client = redis_client
    
    def set_string(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set string value with optional TTL.
        
        Args:
            key: Redis key
            value: String value
            ttl: TTL in seconds (None = no TTL)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if ttl is not None:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)
            return True
        except RedisError as e:
            logger.error(f"Failed to set string {key}: {e}")
            return False
    
    def get_string(self, key: str) -> Optional[str]:
        """Get string value.
        
        Args:
            key: Redis key
        
        Returns:
            String value or None if not found
        """
        try:
            value = self._client.get(key)
            return value.decode('utf-8') if value else None
        except RedisError as e:
            logger.error(f"Failed to get string {key}: {e}")
            return None
    
    def incr(self, key: str) -> Optional[int]:
        """Increment counter.
        
        Args:
            key: Redis key
        
        Returns:
            New counter value or None on error
        """
        try:
            return self._client.incr(key)
        except RedisError as e:
            logger.error(f"Failed to increment {key}: {e}")
            return None
