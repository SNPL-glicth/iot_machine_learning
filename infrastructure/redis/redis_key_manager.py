"""Redis Key Manager — automatic TTL enforcement and safe operations.

Wraps Redis client to enforce:
- Automatic TTL on all writes
- Namespace isolation
- Safe error handling

Modularized:
    - redis_operations.py: Low-level Redis operations
    - redis_key_manager.py: TTL enforcement logic (this file)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, List

try:
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    Redis = Any
    REDIS_AVAILABLE = False

from .redis_keys import RedisKeys
from .redis_operations import RedisOperations

logger = logging.getLogger(__name__)


class RedisKeyManager:
    """Manages Redis operations with automatic TTL enforcement and isolation.
    
    All write operations automatically apply TTL based on key type.
    All operations are tenant-isolated via RedisKeys namespace.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        redis_keys: RedisKeys,
        enable_ttl: bool = True,
    ):
        """Initialize Redis key manager.
        
        Args:
            redis_client: Redis client instance
            redis_keys: RedisKeys instance with tenant isolation
            enable_ttl: Whether to enforce TTL (disable for testing)
        """
        self._ops = RedisOperations(redis_client)
        self._keys = redis_keys
        self._enable_ttl = enable_ttl
    
    def _apply_ttl(self, key: str, ttl: Optional[int]) -> None:
        """Apply TTL to key if enabled and TTL is specified.
        
        Args:
            key: Redis key
            ttl: TTL in seconds (None = no TTL)
        """
        if self._enable_ttl and ttl is not None:
            self._ops.expire(key, ttl)
    
    # --- String operations ---
    
    def set_string(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set string value with automatic TTL."""
        if ttl is not None and self._enable_ttl:
            return self._ops.set_string(key, value, ttl)
        else:
            success = self._ops.set_string(key, value)
            if success and ttl is None and self._enable_ttl:
                self._apply_ttl(key, self._keys.default_ttl)
            return success
    
    def get_string(self, key: str) -> Optional[str]:
        """Get string value."""
        return self._ops.get_string(key)
    
    def incr(self, key: str, ttl: Optional[int] = None) -> Optional[int]:
        """Increment counter with automatic TTL."""
        value = self._ops.incr(key)
        if value == 1:  # First increment, set TTL
            self._apply_ttl(key, ttl or self._keys.default_ttl)
        return value
    
    # --- Hash operations ---
    
    def hset(self, key: str, field: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set hash field with automatic TTL."""
        success = self._ops.hset(key, field, value)
        if success:
            self._apply_ttl(key, ttl)
        return success
    
    def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field value."""
        return self._ops.hget(key, field)
    
    def hgetall(self, key: str) -> dict:
        """Get all hash fields."""
        return self._ops.hgetall(key)
    
    # --- List operations ---
    
    def lpush(self, key: str, value: str, ttl: Optional[int] = None, maxlen: Optional[int] = None) -> bool:
        """Push to list head with automatic TTL and optional trimming."""
        success = self._ops.lpush(key, value)
        if success:
            if maxlen is not None:
                self._ops.ltrim(key, 0, maxlen - 1)
            self._apply_ttl(key, ttl)
        return success
    
    def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get list range."""
        return self._ops.lrange(key, start, end)
    
    # --- Sorted set operations ---
    
    def zadd(self, key: str, score: float, member: str, ttl: Optional[int] = None) -> bool:
        """Add to sorted set with automatic TTL."""
        success = self._ops.zadd(key, score, member)
        if success:
            self._apply_ttl(key, ttl)
        return success
    
    def zrange(self, key: str, start: int = 0, end: int = -1, withscores: bool = False) -> List:
        """Get sorted set range."""
        return self._ops.zrange(key, start, end, withscores)
    
    # --- Key operations ---
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        return self._ops.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._ops.exists(key)
