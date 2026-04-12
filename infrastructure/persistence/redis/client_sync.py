"""Synchronous Redis clients.

Extracted from clients.py as part of modularization.
Provides sync clients for general and stream operations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .pools import get_general_pool, get_stream_pool
from .utils import _detect_async_context

logger = logging.getLogger(__name__)

# Sync client instances
_general_sync_client: Optional[Any] = None
_stream_sync_client: Optional[Any] = None


def get_sync_client() -> Any:
    """Get a synchronous Redis client using the general pool.
    
    SAFETY: Raises RuntimeError if called from async context.
    """
    global _general_sync_client
    
    if _detect_async_context():
        raise RuntimeError(
            "CRITICAL: Sync Redis client used inside async context. "
            "This will block the event loop. "
            "Use await RedisConnectionManager.get_async_client() instead."
        )
    
    if _general_sync_client is not None:
        return _general_sync_client
    
    try:
        import redis
        pool = get_general_pool()
        _general_sync_client = redis.Redis(connection_pool=pool)
        _general_sync_client.ping()
        logger.debug("redis_sync_client_initialized")
        return _general_sync_client
        
    except Exception as e:
        logger.error("redis_sync_client_failed", extra={"error": str(e)})
        raise RuntimeError(f"Redis connection failed: {e}") from e


def get_stream_client() -> Any:
    """Get a synchronous Redis client using the stream pool.
    
    Use this for blocking stream operations (xreadgroup, etc).
    """
    global _stream_sync_client
    
    if _detect_async_context():
        raise RuntimeError(
            "CRITICAL: Stream Redis client used inside async context. "
            "Stream operations block. Use async streams or run in thread."
        )
    
    if _stream_sync_client is not None:
        return _stream_sync_client
    
    try:
        import redis
        pool = get_stream_pool()
        _stream_sync_client = redis.Redis(connection_pool=pool)
        _stream_sync_client.ping()
        logger.debug("redis_stream_client_initialized")
        return _stream_sync_client
        
    except Exception as e:
        logger.error("redis_stream_client_failed", extra={"error": str(e)})
        raise RuntimeError(f"Redis stream connection failed: {e}") from e


def reset_sync_clients() -> None:
    """Reset sync client connections (for testing)."""
    global _general_sync_client, _stream_sync_client
    
    for client in [_general_sync_client, _stream_sync_client]:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    
    _general_sync_client = None
    _stream_sync_client = None
    logger.debug("redis_sync_clients_reset")
