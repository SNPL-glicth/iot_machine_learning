"""Redis client for plasticity — isolated Redis operations.

Fail-safe Redis operations with scope-aware keys.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from iot_machine_learning.domain.value_objects.plasticity_scope import PlasticityScope
from iot_machine_learning.infrastructure.redis_keys import RedisKeys

logger = logging.getLogger(__name__)


class PlasticityRedisClient:
    """Redis operations for plasticity with scope support.
    
    Responsibilities:
    - Scoped key generation
    - Weight reading with local cache
    - Weight writing (single and batch)
    
    All methods are fail-safe: errors are logged but not raised.
    """
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        scope: Optional[PlasticityScope] = None,
        cache_ttl_seconds: float = 60.0,
    ) -> None:
        self._redis = redis_client
        self._scope = scope
        self._cache_ttl = cache_ttl_seconds
        self._local_cache: Dict[str, Tuple[Dict[str, float], float]] = {}
    
    def _get_redis_key(self, regime: str) -> str:
        """Generate Redis key for this regime, respecting scope if set."""
        if self._scope is not None:
            return self._scope.with_regime(regime).redis_key
        return RedisKeys.plasticity(regime)
    
    def get_weights(
        self,
        regime: str,
        engine_names: List[str],
        min_weight: float,
    ) -> Optional[Dict[str, float]]:
        """Fetch weights from Redis with local caching.
        
        Args:
            regime: Current regime label
            engine_names: List of engines to fetch
            min_weight: Default weight for missing engines
            
        Returns:
            Normalized weights dict, or None if Redis unavailable
        """
        if self._redis is None:
            return None
        
        cache_key = self._get_redis_key(regime)
        import time
        now = time.monotonic()
        
        # Check local cache first
        if cache_key in self._local_cache:
            weights, timestamp = self._local_cache[cache_key]
            if now - timestamp < self._cache_ttl:
                return weights
        
        try:
            # Fetch from Redis
            redis_weights = self._redis.hgetall(cache_key)
            if not redis_weights:
                return None
            
            # Parse and filter to requested engines
            weights = {}
            for name in engine_names:
                if name in redis_weights:
                    weights[name] = float(redis_weights[name])
                else:
                    weights[name] = min_weight
            
            # Normalize
            total = sum(weights.values())
            if total < 1e-12:
                return None
            
            normalized = {name: w / total for name, w in weights.items()}
            
            # Cache locally
            self._local_cache[cache_key] = (normalized, now)
            
            return normalized
            
        except Exception as e:
            logger.debug(f"redis_weights_fetch_failed: {e}")
            return None
    
    def update_weight(
        self,
        regime: str,
        engine_name: str,
        accuracy: float,
    ) -> None:
        """Update single engine weight in Redis (fail-safe)."""
        if self._redis is None:
            return
        
        try:
            cache_key = self._get_redis_key(regime)
            self._redis.hset(cache_key, engine_name, str(accuracy))
            # Invalidate local cache to force refresh on next read
            self._local_cache.pop(cache_key, None)
        except Exception as e:
            logger.debug(f"redis_update_failed: {e}")
    
    def update_weights_batch(
        self,
        regime: str,
        engine_accuracies: Dict[str, float],
    ) -> None:
        """Batch update multiple engine weights using Redis pipeline."""
        if self._redis is None:
            return
        
        try:
            cache_key = self._get_redis_key(regime)
            pipe = self._redis.pipeline()
            for engine_name, accuracy in engine_accuracies.items():
                pipe.hset(cache_key, engine_name, str(accuracy))
            pipe.execute()
            # Invalidate local cache
            self._local_cache.pop(cache_key, None)
            logger.debug(f"redis_pipeline_updated: {len(engine_accuracies)} engines for {regime}")
        except Exception as e:
            logger.warning(f"redis_pipeline_failed: {e}")
    
    def invalidate_cache(self, regime: str) -> None:
        """Invalidate local cache for a regime."""
        cache_key = self._get_redis_key(regime)
        self._local_cache.pop(cache_key, None)
