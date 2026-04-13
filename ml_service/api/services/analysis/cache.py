"""Analysis cache implementation with LRU eviction.

Extracted from document_analyzer.py as part of refactoring Paso 1.
Provides content-based caching for document analysis results.

Supports Redis persistence for cache survival across restarts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AnalysisCache:
    """Content-based cache for document analysis results.
    
    Uses LRU eviction when capacity is reached.
    Thread-safe for single-process usage (not multi-process safe).
    
    Args:
        max_entries: Maximum number of entries to store before eviction.
    """
    
    def __init__(self, max_entries: int = 100, redis_client: Optional[Any] = None) -> None:
        """Initialize cache with specified capacity.
        
        Args:
            max_entries: Maximum in-memory cache entries
            redis_client: Optional Redis client for persistent cache
        """
        self._max_entries = max_entries
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._redis = redis_client
        self._redis_available = False
        
        # Test Redis connection
        if self._redis is not None:
            try:
                self._redis.ping()
                self._redis_available = True
                logger.info("analysis_cache_redis_connected")
            except Exception as e:
                logger.warning(f"analysis_cache_redis_unavailable: {e}")
                self._redis_available = False
    
    def compute_content_hash(self, content: str) -> str:
        """Compute MD5 hash of content for cache key."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
    
    def build_cache_key(self, content_hash: str, content_type: str) -> str:
        """Build cache key from content hash and type."""
        return f"{content_hash}:{content_type}"
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if exists.
        
        Args:
            cache_key: Key to look up.
            
        Returns:
            Cached result dict or None if not found.
        """
        # Try in-memory cache first
        result = self._cache.get(cache_key)
        if result is not None:
            return result
        
        # Try Redis if available
        if self._redis_available:
            try:
                redis_key = f"zenin:analysis:{cache_key}"
                cached_json = self._redis.get(redis_key)
                if cached_json:
                    result = json.loads(cached_json)
                    # Populate in-memory cache
                    self._cache[cache_key] = result
                    logger.debug(f"analysis_cache_redis_hit: {cache_key[:20]}...")
                    return result
            except Exception as e:
                logger.warning(f"analysis_cache_redis_get_failed: {e}")
        
        return None
    
    def set(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Store result in cache with LRU eviction.
        
        Args:
            cache_key: Key for the result.
            result: Result dict to cache.
        """
        # Evict oldest if at capacity (simple LRU: clear half if full)
        if len(self._cache) >= self._max_entries:
            # Remove oldest 50% of entries
            keys_to_remove = list(self._cache.keys())[: self._max_entries // 2]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"analysis_cache_evicted: removed {len(keys_to_remove)} entries")
        
        # Store in memory
        self._cache[cache_key] = result
        logger.debug(
            f"analysis_cache_stored: key={cache_key[:20]}...",
            extra={"cache_size": len(self._cache)},
        )
        
        # Store in Redis if available
        if self._redis_available:
            try:
                redis_key = f"zenin:analysis:{cache_key}"
                result_json = json.dumps(result, default=str)  # default=str for non-serializable types
                self._redis.setex(redis_key, 3600, result_json)  # TTL = 3600 seconds (1 hour)
                logger.debug(f"analysis_cache_redis_stored: {cache_key[:20]}...")
            except Exception as e:
                logger.warning(f"analysis_cache_redis_set_failed: {e}")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("analysis_cache_cleared")
    
    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)
    
    @property
    def max_entries(self) -> int:
        """Maximum cache capacity."""
        return self._max_entries
