"""Flag Cache with TTL (COG-SEV-3).

Caches feature flags to avoid reading on every decision call.

Applies SRP: FlagCache only manages caching, not flag logic.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from iot_machine_learning.ml_service.config.flags import FeatureFlags

logger = logging.getLogger(__name__)


class FlagCache:
    """Thread-safe cache for feature flags with TTL (COG-SEV-3).
    
    Reduces overhead of reading flags on every decision call.
    
    Attributes:
        _ttl_seconds: Time-to-live for cached flags.
        _cached_flags: Cached FeatureFlags instance.
        _cache_timestamp: Timestamp when cache was last updated.
        _lock: Thread lock for concurrent access.
    
    Applies SRP: Only caches flags, doesn't implement flag logic.
    Thread-safe: Uses threading.Lock for concurrent access.
    """
    
    def __init__(self, ttl_seconds: float = 60.0) -> None:
        """Initialize flag cache.
        
        Args:
            ttl_seconds: Cache TTL in seconds (default: 60s).
        """
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        
        self._ttl_seconds = ttl_seconds
        self._cached_flags: Optional[FeatureFlags] = None
        self._cache_timestamp: float = 0.0
        self._lock = threading.Lock()
    
    def get_or_load(self, loader_fn: callable) -> FeatureFlags:
        """Get cached flags or reload if expired.
        
        Args:
            loader_fn: Function to load flags when cache expired.
                Should return FeatureFlags instance.
        
        Returns:
            FeatureFlags instance (cached or freshly loaded).
        
        Thread-safe: Uses lock to prevent concurrent reloads.
        """
        with self._lock:
            now = time.monotonic()
            
            # Check if cache is valid
            if self._cached_flags is not None:
                age = now - self._cache_timestamp
                if age < self._ttl_seconds:
                    logger.debug(
                        "flag_cache_hit",
                        extra={
                            "age_seconds": round(age, 2),
                            "ttl_seconds": self._ttl_seconds,
                        },
                    )
                    return self._cached_flags
            
            # Cache expired or empty, reload
            logger.info(
                "flag_cache_reload",
                extra={
                    "ttl_seconds": self._ttl_seconds,
                    "reason": "expired" if self._cached_flags else "empty",
                },
            )
            
            try:
                self._cached_flags = loader_fn()
                self._cache_timestamp = now
                return self._cached_flags
            except Exception as exc:
                logger.error(
                    "flag_cache_reload_failed",
                    extra={"error": str(exc)},
                )
                # Return stale cache if available, otherwise re-raise
                if self._cached_flags is not None:
                    logger.warning(
                        "flag_cache_using_stale",
                        extra={"age_seconds": round(now - self._cache_timestamp, 2)},
                    )
                    return self._cached_flags
                raise
    
    def invalidate(self) -> None:
        """Invalidate cache, forcing reload on next access.
        
        Thread-safe.
        """
        with self._lock:
            self._cached_flags = None
            self._cache_timestamp = 0.0
            logger.info("flag_cache_invalidated")
    
    def get_metrics(self) -> dict:
        """Get cache metrics.
        
        Returns:
            Dict with cache state and metrics.
        
        Thread-safe.
        """
        with self._lock:
            now = time.monotonic()
            age = now - self._cache_timestamp if self._cached_flags else None
            
            return {
                "ttl_seconds": self._ttl_seconds,
                "is_cached": self._cached_flags is not None,
                "age_seconds": round(age, 2) if age is not None else None,
                "is_expired": age >= self._ttl_seconds if age is not None else None,
            }
