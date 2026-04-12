"""Redis connection manager — Re-export from modular redis package.

DEPRECATED: This module is kept for backward compatibility.
Import from the modular package instead:
    from iot_machine_learning.infrastructure.persistence.redis import (
        RedisConnectionManager,
        get_sync_redis,
        get_async_redis,
    )

The implementation has been split into focused modules:
- redis.config: Settings and constants
- redis.pools: Connection pool management
- redis.clients: Client creation and access
- redis.utils: Helper functions
"""

from __future__ import annotations

import warnings

warnings.warn(
    "redis_connection_manager.py is deprecated. "
    "Import from infrastructure.persistence.redis instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all public APIs from modular package
from .redis import (
    RedisConnectionManager,
    get_sync_redis,
    get_async_redis,
    get_redis_pool,
    DEFAULT_REDIS_URL,
    GENERAL_MAX_CONNECTIONS,
    STREAM_MAX_CONNECTIONS,
)

__all__ = [
    "RedisConnectionManager",
    "get_sync_redis",
    "get_async_redis",
    "get_redis_pool",
    "DEFAULT_REDIS_URL",
    "GENERAL_MAX_CONNECTIONS",
    "STREAM_MAX_CONNECTIONS",
]
