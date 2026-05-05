"""Redis key management and operations package.

Centralized Redis key patterns, operations, and management with:
- Strict namespace isolation (env:app:tenant:resource)
- Automatic TTL enforcement
- Safe error handling

Modules:
    redis_keys_base: Core sanitization and key building
    redis_keys_registry: Specific key patterns and TTL definitions
    redis_keys: Facade exposing RedisKeys class
    redis_operations: Facade for all Redis operations
    redis_ops_string: String and counter operations
    redis_ops_collection: Hash, list, and sorted set operations
    redis_ops_key: Key management operations
    redis_key_manager: TTL enforcement and safe operations wrapper
"""

from __future__ import annotations

from .redis_keys import RedisKeys
from .redis_key_manager import RedisKeyManager

__all__ = ["RedisKeys", "RedisKeyManager"]
