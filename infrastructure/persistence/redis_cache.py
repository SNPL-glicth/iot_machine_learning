"""Cache Redis canónico con TTL automático. Implementa CachePort.

FIX 2026-04-09:
- Circuit breaker protection for graceful degradation
- Better error handling with fallback

# Reemplaza proxy de 29 líneas — ver cache_system/ para historial
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Any
from redis import Redis, ConnectionError as RedisConnectionError

from iot_machine_learning.domain.ports.document_analysis import CachePort, AnalysisOutput
from iot_machine_learning.infrastructure.persistence.redis import (
    get_redis_circuit_breaker,
)

logger = logging.getLogger(__name__)
_PREFIX = "zenin:analysis:"
_DEFAULT_TTL = 300  # 5 minutos


class RedisAnalysisCache(CachePort):
    """Cache de análisis respaldado por Redis.

    Implementa CachePort definido en T2.
    Reemplaza InMemoryAnalysisCache en producción.
    Fallback automático a no-cache si Redis no está disponible.
    
    FIX 2026-04-09: Added circuit breaker for resilience.
    """

    def __init__(
        self,
        redis_client: Redis,
        ttl_seconds: int = _DEFAULT_TTL,
        key_prefix: str = _PREFIX,
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._prefix = key_prefix
        
        # Circuit breaker for cache operations
        self._circuit_breaker = get_redis_circuit_breaker("redis_cache")

    def _get_sync(self, key: str) -> Optional[AnalysisOutput]:
        """Synchronous get implementation."""
        raw = self._redis.get(f"{self._prefix}{key}")
        if raw is None:
            return None
        data = json.loads(raw)
        return AnalysisOutput(**data)
    
    def get(self, key: str) -> Optional[AnalysisOutput]:
        """Obtiene resultado del cache. Retorna None si no existe o Redis falla.
        
        Uses circuit breaker for graceful degradation.
        """
        def _fallback():
            logger.debug("Cache get bypassed (circuit open): %s", key[:20])
            return None
        
        try:
            return self._circuit_breaker.call(
                lambda: self._get_sync(key),
                _fallback
            )
        except RedisConnectionError:
            logger.warning("Redis no disponible en get(%s), cache miss.", key)
            return None
        except Exception as exc:
            logger.error("Error inesperado en cache.get: %s", exc)
            return None

    def _set_sync(
        self,
        key: str,
        value: AnalysisOutput,
        ttl: Optional[int] = None,
    ) -> None:
        """Synchronous set implementation."""
        # AnalysisOutput es dataclass — usar __dict__ para serialización
        payload = json.dumps(value.__dict__)
        self._redis.setex(
            f"{self._prefix}{key}",
            ttl or self._ttl,
            payload,
        )
    
    def set(
        self,
        key: str,
        value: AnalysisOutput,
        ttl: Optional[int] = None,
    ) -> None:
        """Guarda en cache con TTL. Silencia fallos de Redis.
        
        Uses circuit breaker for graceful degradation.
        """
        def _fallback():
            logger.debug("Cache set bypassed (circuit open): %s", key[:20])
            return None
        
        try:
            self._circuit_breaker.call(
                lambda: self._set_sync(key, value, ttl),
                _fallback
            )
        except RedisConnectionError:
            logger.warning("Redis no disponible en set(%s), omitiendo cache.", key)
        except Exception as exc:
            logger.error("Error inesperado en cache.set: %s", exc)

    def invalidate(self, key: str) -> None:
        """Elimina entrada del cache."""
        try:
            self._redis.delete(f"{self._prefix}{key}")
        except RedisConnectionError:
            logger.warning("Redis no disponible en invalidate(%s).", key)

    def health_check(self) -> bool:
        """Retorna True si Redis responde correctamente."""
        try:
            return bool(self._redis.ping())
        except RedisConnectionError:
            return False
