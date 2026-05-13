"""Cached posterior storage with write-through (PERF-CRIT-2).

Combines Redis backend with local cache for performance.

Applies SRP: Storage orchestration separate from cache and Redis logic.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .storage_interface import IPosteriorStorage
from .posterior_cache import PosteriorCache


class CachedPosteriorStorage(IPosteriorStorage):
    """Cached storage with write-through to Redis (PERF-CRIT-2).
    
    Reads: Check cache first, fallback to Redis, populate cache.
    Writes: Update Redis AND cache simultaneously (write-through).
    
    Attributes:
        _redis_storage: Backend Redis storage.
        _cache: Local in-memory cache with TTL.
    
    Applies SRP: Orchestrates cache + Redis, doesn't implement either.
    Applies DIP: Depends on IPosteriorStorage for Redis backend.
    """
    
    def __init__(
        self,
        redis_storage: IPosteriorStorage,
        cache_ttl_seconds: float = 60.0,
    ) -> None:
        """Initialize cached storage.
        
        Args:
            redis_storage: Backend Redis storage implementation.
            cache_ttl_seconds: Cache TTL in seconds.
        """
        self._redis_storage = redis_storage
        self._cache = PosteriorCache(ttl_seconds=cache_ttl_seconds)
    
    def load_posterior(
        self,
        regime: str,
        engine_name: str,
    ) -> Optional[Tuple[float, float]]:
        """Load posterior with cache-aside pattern.
        
        1. Check cache
        2. If miss, load from Redis
        3. Populate cache
        
        Args:
            regime: Regime key.
            engine_name: Engine name.
        
        Returns:
            Tuple of (mu, sigma2) if exists, else None.
        """
        # Check cache first
        cached = self._cache.get(regime, engine_name)
        if cached is not None:
            return cached
        
        # Cache miss: load from Redis
        posterior = self._redis_storage.load_posterior(regime, engine_name)
        
        if posterior is not None:
            # Populate cache
            mu, sigma2 = posterior
            self._cache.put(regime, engine_name, mu, sigma2)
        
        return posterior
    
    def save_posterior(
        self,
        regime: str,
        engine_name: str,
        mu: float,
        sigma2: float,
    ) -> None:
        """Save posterior with write-through.
        
        Updates Redis AND cache simultaneously.
        
        Args:
            regime: Regime key.
            engine_name: Engine name.
            mu: Posterior mean.
            sigma2: Posterior variance.
        """
        # Write-through: update both
        self._redis_storage.save_posterior(regime, engine_name, mu, sigma2)
        self._cache.put(regime, engine_name, mu, sigma2)
    
    def delete_posterior(
        self,
        regime: str,
        engine_name: str,
    ) -> None:
        """Delete posterior from Redis and invalidate cache.
        
        Args:
            regime: Regime key.
            engine_name: Engine name.
        """
        self._redis_storage.delete_posterior(regime, engine_name)
        self._cache.invalidate(regime, engine_name)
    
    def evict_expired_cache_entries(self) -> int:
        """Evict expired cache entries.
        
        Returns:
            Number of entries evicted.
        """
        return self._cache.evict_expired()
    
    def clear_cache(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
    
    def cache_size(self) -> int:
        """Get cache size.
        
        Returns:
            Number of cached entries.
        """
        return self._cache.size()
