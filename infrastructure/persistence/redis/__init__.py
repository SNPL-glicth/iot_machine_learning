"""Redis persistence layer — Modular package.

FIX 2026-04-09: Split from monolithic redis_connection_manager.py (>180 lines)
into focused modules (<180 lines each):
- config.py: Settings (17 lines)
- utils.py: Helper functions (31 lines)
- pools.py: Connection pools (170 lines)
- client_sync.py: Sync clients (95 lines)
- client_async.py: Async client (54 lines)
- clients.py: Manager facade (117 lines)
- circuit_breaker.py: Circuit breaker class (144 lines)
- circuit_factory.py: Circuit factory (68 lines)

Public API exported here for backward compatibility.
"""

from __future__ import annotations

from .config import (
    DEFAULT_REDIS_URL,
    DEFAULT_SOCKET_TIMEOUT,
    GENERAL_MAX_CONNECTIONS,
    STREAM_MAX_CONNECTIONS,
)
from .utils import _detect_async_context, _get_redis_url
from .pools import (
    get_general_pool,
    get_stream_pool,
    get_async_pool,
)
from .clients import (
    RedisConnectionManager,
    get_sync_redis,
    get_async_redis,
    get_redis_pool,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
)
from .circuit_factory import (
    get_redis_circuit_breaker,
    reset_all_circuits,
)

__all__ = [
    # Main class
    "RedisConnectionManager",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "get_redis_circuit_breaker",
    "reset_all_circuits",
    # Config
    "DEFAULT_REDIS_URL",
    "DEFAULT_SOCKET_TIMEOUT",
    "GENERAL_MAX_CONNECTIONS",
    "STREAM_MAX_CONNECTIONS",
    # Functions
    "get_sync_redis",
    "get_async_redis",
    "get_redis_pool",
    "get_general_pool",
    "get_stream_pool",
    "get_async_pool",
    "_detect_async_context",
]
