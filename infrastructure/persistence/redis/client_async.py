"""Asynchronous Redis client.

Extracted from clients.py as part of modularization.
Provides async client for non-blocking operations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .pools import get_async_pool

logger = logging.getLogger(__name__)

# Async client instance
_async_client: Optional[Any] = None


async def get_async_client() -> Any:
    """Get an asynchronous Redis client.
    
    SAFE to call from any async context.
    
    Returns:
        redis.asyncio.Redis instance
        
    Raises:
        RuntimeError: If Redis is not available
    """
    global _async_client
    
    if _async_client is not None:
        return _async_client
    
    try:
        import redis.asyncio as aioredis
        pool = get_async_pool()
        _async_client = aioredis.Redis(connection_pool=pool)
        await _async_client.ping()
        logger.debug("redis_async_client_initialized")
        return _async_client
        
    except Exception as e:
        logger.error("redis_async_client_failed", extra={"error": str(e)})
        raise RuntimeError(f"Redis async connection failed: {e}") from e


def reset_async_client() -> None:
    """Reset async client connection (for testing)."""
    global _async_client
    _async_client = None
    logger.debug("redis_async_client_reset")
