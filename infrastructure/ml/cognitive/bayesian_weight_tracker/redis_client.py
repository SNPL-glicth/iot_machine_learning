"""Redis client for Bayesian weight tracker — isolated Redis operations.

Fail-safe Redis operations with scope-aware keys.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from iot_machine_learning.domain.value_objects.plasticity_scope import PlasticityScope
from iot_machine_learning.infrastructure.redis.redis_keys import RedisKeys
from iot_machine_learning.infrastructure.security.redis_namespace import (
    RedisNamespace,
    get_namespace,
)

logger = logging.getLogger(__name__)


class WeightTrackerRedisClient:
    """Redis operations for Bayesian weight tracking with scope support.
    
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
        redis_ttl_seconds: float = 86400.0,
        tenant_id: str = "default",
        namespace: Optional[RedisNamespace] = None,
    ) -> None:
        self._redis = redis_client
        self._scope = scope
        self._cache_ttl = cache_ttl_seconds
        self._redis_ttl = int(redis_ttl_seconds)  # TTL for Redis keys (24h default)
        self._local_cache: Dict[str, Tuple[Dict[str, float], float]] = {}
        
        # Namespace for tenant isolation (SEC-2 fix)
        self._namespace = namespace or get_namespace(tenant_id=tenant_id)
        self._tenant_id = tenant_id
    
    def _get_redis_key(self, regime: str) -> str:
        """Generate Redis key for this regime with namespace.
        
        Uses RedisNamespace for tenant isolation.
        Format: {env}:{app}:{tenant}:weights:{regime}
        """
        if self._scope is not None:
            # Legacy scope support (deprecated path)
            logger.debug(
                "weight_tracker_using_legacy_scope",
                extra={"regime": regime, "scope": str(self._scope)}
            )
            return self._scope.with_regime(regime).redis_key
        
        # New namespaced path (SEC-2 compliant)
        return self._namespace.key(
            resource_type="weights",
            resource_id=regime,
        )
    
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
        """Update single engine weight in Redis (fail-safe).
        
        Sets TTL to prevent memory leaks.
        Uses namespaced keys for tenant isolation.
        """
        if self._redis is None:
            return
        
        try:
            cache_key = self._get_redis_key(regime)
            self._redis.hset(cache_key, engine_name, str(accuracy))
            # Set TTL to prevent memory leak (SEC-1 fix)
            self._redis.expire(cache_key, self._redis_ttl)
            # Invalidate local cache to force refresh on next read
            self._local_cache.pop(cache_key, None)
            
            logger.debug(
                "redis_weight_updated",
                extra={
                    "regime": regime,
                    "engine": engine_name,
                    "key": cache_key,
                    "ttl": self._redis_ttl,
                }
            )
        except Exception as e:
            logger.debug(f"redis_update_failed: {e}")
    
    def update_weights_batch(
        self,
        regime: str,
        engine_accuracies: Dict[str, float],
    ) -> None:
        """Batch update multiple engine weights using Redis pipeline.
        
        Sets TTL to prevent memory leaks.
        Uses namespaced keys for tenant isolation.
        """
        if self._redis is None:
            return
        
        try:
            cache_key = self._get_redis_key(regime)
            pipe = self._redis.pipeline()
            for engine_name, accuracy in engine_accuracies.items():
                pipe.hset(cache_key, engine_name, str(accuracy))
            # Set TTL in pipeline (SEC-1 fix)
            pipe.expire(cache_key, self._redis_ttl)
            pipe.execute()
            # Invalidate local cache
            self._local_cache.pop(cache_key, None)
            
            logger.debug(
                "redis_pipeline_updated",
                extra={
                    "regime": regime,
                    "n_engines": len(engine_accuracies),
                    "key": cache_key,
                    "ttl": self._redis_ttl,
                }
            )
        except Exception as e:
            logger.warning(f"redis_pipeline_failed: {e}")
    
    def invalidate_cache(self, regime: str) -> None:
        """Invalidate local cache for a regime."""
        cache_key = self._get_redis_key(regime)
        self._local_cache.pop(cache_key, None)
