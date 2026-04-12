"""Factory de cache. Selecciona Redis o InMemory según entorno.

MIGRATED 2026-04-09: Now uses RedisConnectionManager for centralized connection.
"""
from __future__ import annotations

import os
import logging

from iot_machine_learning.domain.ports.document_analysis import CachePort
from iot_machine_learning.infrastructure.persistence.redis import (
    RedisConnectionManager,
)

logger = logging.getLogger(__name__)


def build_analysis_cache() -> CachePort:
    """Retorna RedisAnalysisCache si REDIS_URL está definida, InMemory si no.

    Si Redis está configurado pero no conecta, hace fallback a InMemory
    y loguea un warning. Nunca lanza excepción.
    
    MIGRATED: Now uses RedisConnectionManager for connection pooling.
    """
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            from iot_machine_learning.infrastructure.persistence.redis_cache import RedisAnalysisCache
            # Use centralized connection manager
            client = RedisConnectionManager.get_sync_client()
            logger.info("Cache: Redis activo via RedisConnectionManager")
            return RedisAnalysisCache(client)
        except Exception as exc:
            logger.warning("Redis no disponible (%s) — usando InMemory.", exc)

    from iot_machine_learning.infrastructure.persistence.cache import InMemoryAnalysisCache
    logger.info("Cache: InMemoryAnalysisCache activo")
    return InMemoryAnalysisCache()


# Helper para compute_content_hash (re-exportado de cache.py para compatibilidad)
from iot_machine_learning.infrastructure.persistence.cache import (
    compute_content_hash,
    build_cache_key,
)

__all__ = ["build_analysis_cache", "compute_content_hash", "build_cache_key"]
