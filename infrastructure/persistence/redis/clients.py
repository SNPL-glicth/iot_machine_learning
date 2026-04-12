"""Redis connection manager — Thin wrapper around specialized modules.

This module re-exports from focused modules:
- client_sync: Synchronous clients (general + stream)
- client_async: Asynchronous client
- pools: Connection pool management
- utils: Helper functions

Provides unified RedisConnectionManager class for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from .pools import get_general_pool, get_stream_pool
from .client_sync import get_sync_client, get_stream_client, reset_sync_clients
from .client_async import get_async_client, reset_async_client
from .utils import _get_redis_url


class RedisConnectionManager:
    """Singleton Redis connection manager — delegates to focused modules."""
    
    @classmethod
    def get_pool(cls) -> Any:
        """Get the general connection pool (legacy compatibility)."""
        return get_general_pool()
    
    @classmethod
    def get_general_pool(cls) -> Any:
        """Get the general operations connection pool."""
        return get_general_pool()
    
    @classmethod
    def get_stream_pool(cls) -> Any:
        """Get the stream operations connection pool."""
        return get_stream_pool()
    
    @classmethod
    def get_sync_client(cls) -> Any:
        """Get a synchronous Redis client using the general pool."""
        return get_sync_client()
    
    @classmethod
    def get_stream_client(cls) -> Any:
        """Get a synchronous Redis client using the stream pool."""
        return get_stream_client()
    
    @classmethod
    async def get_async_client(cls) -> Any:
        """Get an asynchronous Redis client."""
        return await get_async_client()
    
    @classmethod
    def reset(cls) -> None:
        """Reset all connections (for testing)."""
        reset_sync_clients()
        reset_async_client()
        from .pools import reset_pools
        reset_pools()
    
    @classmethod
    def health_check(cls) -> dict:
        """Check Redis health status."""
        from .pools import _general_pool, _stream_pool, _async_pool
        
        result = {
            "general_pool_initialized": _general_pool is not None,
            "stream_pool_initialized": _stream_pool is not None,
            "async_pool_initialized": _async_pool is not None,
            "ping_ok": False,
        }
        
        try:
            import redis
            test_client = redis.Redis(connection_pool=get_general_pool())
            test_client.ping()
            result["ping_ok"] = True
        except Exception as e:
            result["ping_error"] = str(e)
        
        return result
    
    @classmethod
    def get_pool_metrics(cls) -> dict:
        """Get connection pool metrics."""
        from .pools import _general_pool, _stream_pool
        
        metrics = {"general_pool": None, "stream_pool": None}
        
        for pool, name in [(_general_pool, "general_pool"), (_stream_pool, "stream_pool")]:
            if pool is not None:
                metrics[name] = {
                    "max": pool.max_connections,
                    "in_use": len(getattr(pool, '_in_use_connections', set())),
                    "available": len(getattr(pool, '_available_connections', [])),
                }
        
        return metrics


# Convenience functions
def get_sync_redis() -> Any:
    """Get synchronous Redis client."""
    return get_sync_client()


async def get_async_redis() -> Any:
    """Get asynchronous Redis client."""
    return await get_async_client()


def get_redis_pool() -> Any:
    """Get the shared connection pool."""
    return get_general_pool()
