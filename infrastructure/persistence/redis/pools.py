"""Redis connection pool management.

Extracted from redis_connection_manager.py as part of modularization.
Manages separate pools for general, stream, and async operations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .config import (
    GENERAL_MAX_CONNECTIONS,
    STREAM_MAX_CONNECTIONS,
    DEFAULT_SOCKET_TIMEOUT,
    DEFAULT_SOCKET_CONNECT_TIMEOUT,
)
from .utils import _get_redis_url

logger = logging.getLogger(__name__)

# Global connection pools (lazy-initialized)
_general_pool: Optional[Any] = None
_stream_pool: Optional[Any] = None
_async_pool: Optional[Any] = None


def get_general_pool() -> Any:
    """Create/get connection pool for general operations (cache, ML, API)."""
    global _general_pool
    
    if _general_pool is not None:
        return _general_pool
    
    try:
        import redis
        from redis.connection import ConnectionPool
        
        redis_url = _get_redis_url()
        connection_kwargs = redis.connection.parse_url(redis_url)
        
        _general_pool = ConnectionPool(
            host=connection_kwargs.get("host", "localhost"),
            port=connection_kwargs.get("port", 6379),
            db=connection_kwargs.get("db", 0),
            password=connection_kwargs.get("password"),
            username=connection_kwargs.get("username"),
            socket_timeout=2.0,  # Shorter timeout for fast operations
            socket_connect_timeout=5.0,
            max_connections=GENERAL_MAX_CONNECTIONS,
            decode_responses=False,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        
        logger.info(
            "redis_general_pool_created",
            extra={
                "host": connection_kwargs.get("host"),
                "port": connection_kwargs.get("port"),
                "max_connections": GENERAL_MAX_CONNECTIONS,
            }
        )
        return _general_pool
        
    except Exception as e:
        logger.error(
            "redis_general_pool_failed",
            extra={"error": str(e), "url": _get_redis_url().split("@")[-1]}
        )
        raise


def get_stream_pool() -> Any:
    """Create/get dedicated pool for stream operations (blocking reads)."""
    global _stream_pool
    
    if _stream_pool is not None:
        return _stream_pool
    
    try:
        import redis
        from redis.connection import ConnectionPool
        
        redis_url = _get_redis_url()
        connection_kwargs = redis.connection.parse_url(redis_url)
        
        _stream_pool = ConnectionPool(
            host=connection_kwargs.get("host", "localhost"),
            port=connection_kwargs.get("port", 6379),
            db=connection_kwargs.get("db", 0),
            password=connection_kwargs.get("password"),
            username=connection_kwargs.get("username"),
            socket_timeout=None,  # No timeout for blocking operations
            socket_connect_timeout=10.0,
            max_connections=STREAM_MAX_CONNECTIONS,
            decode_responses=False,
            retry_on_timeout=False,  # Don't retry blocking ops
            health_check_interval=60,
        )
        
        logger.info(
            "redis_stream_pool_created",
            extra={
                "host": connection_kwargs.get("host"),
                "port": connection_kwargs.get("port"),
                "max_connections": STREAM_MAX_CONNECTIONS,
            }
        )
        return _stream_pool
        
    except Exception as e:
        logger.error(
            "redis_stream_pool_failed",
            extra={"error": str(e), "url": _get_redis_url().split("@")[-1]}
        )
        raise


def get_async_pool() -> Any:
    """Create/get async connection pool."""
    global _async_pool
    
    if _async_pool is not None:
        return _async_pool
    
    try:
        import redis.asyncio as aioredis
        from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool
        
        redis_url = _get_redis_url()
        
        _async_pool = AsyncConnectionPool.from_url(
            redis_url,
            max_connections=GENERAL_MAX_CONNECTIONS,
            socket_timeout=2.0,
            socket_connect_timeout=5.0,
            decode_responses=False,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        
        logger.info("redis_async_pool_created")
        return _async_pool
        
    except Exception as e:
        logger.error(
            "redis_async_pool_failed",
            extra={"error": str(e)}
        )
        raise


def reset_pools() -> None:
    """Reset all connection pools (for testing)."""
    global _general_pool, _stream_pool, _async_pool
    
    for pool, name in [(_general_pool, "general"), (_stream_pool, "stream")]:
        if pool is not None:
            try:
                pool.disconnect()
            except Exception:
                pass
    
    _general_pool = None
    _stream_pool = None
    _async_pool = None
    
    logger.debug("redis_pools_reset")
