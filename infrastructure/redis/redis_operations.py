"""Redis Operations — facade for all Redis operation modules.

Modularized:
    - redis_ops_string.py: String and counter operations
    - redis_ops_collection.py: Hash, list, and sorted set operations
    - redis_ops_key.py: Key management operations
    - redis_operations.py: Facade combining all (this file)
"""

from __future__ import annotations

from .redis_ops_string import RedisStringOps
from .redis_ops_collection import RedisHashOps, RedisListOps, RedisSortedSetOps
from .redis_ops_key import RedisKeyOps


class RedisOperations:
    """Facade combining all Redis operation classes.
    
    Provides unified interface for all Redis operations while delegating
    to specialized modules. Maintains backward compatibility.
    """
    
    def __init__(self, redis_client):
        """Initialize all operation handlers.
        
        Args:
            redis_client: Redis client instance
        """
        self._string = RedisStringOps(redis_client)
        self._hash = RedisHashOps(redis_client)
        self._list = RedisListOps(redis_client)
        self._sorted_set = RedisSortedSetOps(redis_client)
        self._key = RedisKeyOps(redis_client)
    
    # --- String operations (delegated) ---
    def set_string(self, key, value, ttl=None): return self._string.set_string(key, value, ttl)
    def get_string(self, key): return self._string.get_string(key)
    def incr(self, key): return self._string.incr(key)
    
    # --- Hash operations (delegated) ---
    def hset(self, key, field, value): return self._hash.hset(key, field, value)
    def hget(self, key, field): return self._hash.hget(key, field)
    def hgetall(self, key): return self._hash.hgetall(key)
    
    # --- List operations (delegated) ---
    def lpush(self, key, value): return self._list.lpush(key, value)
    def ltrim(self, key, start, end): return self._list.ltrim(key, start, end)
    def lrange(self, key, start=0, end=-1): return self._list.lrange(key, start, end)
    
    # --- Sorted set operations (delegated) ---
    def zadd(self, key, score, member): return self._sorted_set.zadd(key, score, member)
    def zrange(self, key, start=0, end=-1, withscores=False): return self._sorted_set.zrange(key, start, end, withscores)
    
    # --- Key operations (delegated) ---
    def expire(self, key, ttl): return self._key.expire(key, ttl)
    def delete(self, key): return self._key.delete(key)
    def exists(self, key): return self._key.exists(key)
