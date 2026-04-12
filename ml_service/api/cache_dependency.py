"""FastAPI dependency for CacheManager injection.

Provides singleton CacheManager instance for all routes.

MIGRATED 2026-04-09: Now uses RedisConnectionManager for centralized connection.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends

from iot_machine_learning.infrastructure.persistence.redis import (
    RedisConnectionManager,
)
from iot_machine_learning.infrastructure.persistence.cache_manager import (
    CacheManager,
    CacheTTL,
)

logger = logging.getLogger(__name__)

# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """FastAPI dependency to inject CacheManager.
    
    Returns:
        Singleton CacheManager instance
        
    Raises:
        RuntimeError: If Redis is not available (MANDATORY)
    """
    global _cache_manager
    
    if _cache_manager is not None:
        return _cache_manager
    
    # Initialize Redis client via centralized connection manager
    redis_client = await RedisConnectionManager.get_async_client()
    
    # Create CacheManager
    _cache_manager = CacheManager(
        redis_client=redis_client,
        default_ttl=CacheTTL.ANALYSIS_SHORT,
        key_prefix="zenin:cache:",
    )
    
    logger.info("cache_manager_dependency_initialized")
    
    return _cache_manager


# Type alias for dependency injection
CacheManagerDep = Depends(get_cache_manager)
