"""Posterior cache for BayesianWeightTracker (PERF-CRIT-2).

In-memory cache with TTL to reduce Redis round-trips.

Applies SRP: Cache management is separate concern from Bayesian updates.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class CachedPosterior:
    """Cached posterior parameters with timestamp.
    
    Attributes:
        mu: Posterior mean (accuracy).
        sigma2: Posterior variance.
        timestamp: Cache entry creation time (monotonic).
    """
    mu: float
    sigma2: float
    timestamp: float


class PosteriorCache:
    """Thread-safe in-memory cache for posterior parameters (PERF-CRIT-2).
    
    Reduces Redis round-trips by caching posteriors locally with TTL.
    
    Attributes:
        _cache: Dict mapping (regime, engine_name) → CachedPosterior.
        _ttl_seconds: Time-to-live for cache entries.
        _lock: Thread lock for concurrent access.
    
    Applies SRP: Only manages cache, no Bayesian logic.
    """
    
    def __init__(self, ttl_seconds: float = 60.0) -> None:
        """Initialize cache.
        
        Args:
            ttl_seconds: Cache entry TTL in seconds.
        """
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        
        self._cache: Dict[Tuple[str, str], CachedPosterior] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
    
    def get(
        self,
        regime: str,
        engine_name: str,
    ) -> Optional[Tuple[float, float]]:
        """Get cached posterior if not expired.
        
        Args:
            regime: Regime key.
            engine_name: Engine name.
        
        Returns:
            Tuple of (mu, sigma2) if cached and fresh, else None.
        
        Thread-safe.
        """
        with self._lock:
            key = (regime, engine_name)
            cached = self._cache.get(key)
            
            if cached is None:
                return None
            
            # Check TTL
            now = time.monotonic()
            age = now - cached.timestamp
            
            if age > self._ttl_seconds:
                # Expired, remove
                del self._cache[key]
                return None
            
            return (cached.mu, cached.sigma2)
    
    def put(
        self,
        regime: str,
        engine_name: str,
        mu: float,
        sigma2: float,
    ) -> None:
        """Store posterior in cache.
        
        Args:
            regime: Regime key.
            engine_name: Engine name.
            mu: Posterior mean.
            sigma2: Posterior variance.
        
        Thread-safe.
        """
        with self._lock:
            key = (regime, engine_name)
            self._cache[key] = CachedPosterior(
                mu=mu,
                sigma2=sigma2,
                timestamp=time.monotonic(),
            )
    
    def invalidate(
        self,
        regime: str,
        engine_name: str,
    ) -> None:
        """Invalidate cache entry.
        
        Args:
            regime: Regime key.
            engine_name: Engine name.
        
        Thread-safe.
        """
        with self._lock:
            key = (regime, engine_name)
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear entire cache.
        
        Thread-safe.
        """
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get cache size.
        
        Returns:
            Number of cached entries.
        
        Thread-safe.
        """
        with self._lock:
            return len(self._cache)
    
    def evict_expired(self) -> int:
        """Evict expired entries.
        
        Returns:
            Number of entries evicted.
        
        Thread-safe.
        """
        with self._lock:
            now = time.monotonic()
            expired_keys = [
                key for key, cached in self._cache.items()
                if (now - cached.timestamp) > self._ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
