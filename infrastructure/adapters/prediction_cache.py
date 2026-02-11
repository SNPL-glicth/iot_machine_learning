"""Cache de predicciones con invalidación inteligente.

Implementación en memoria (sin dependencia de Redis) para Fase Enterprise.
Si Redis está disponible, se puede extender con RedisCache.

Estrategia:
- Key: hash(sensor_id + últimos N valores + engine_name)
- TTL: configurable (default 60s)
- Invalidación: si nuevo valor difiere > threshold del último, invalidar
- LRU eviction: máximo de entradas configurable

Performance:
- Hit rate esperado: 40-60% (lecturas frecuentes del mismo sensor)
- Latencia hit: <1ms (in-memory) / <5ms (Redis)
- Latencia miss: 0ms overhead

Thread-safe: usa threading.Lock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrada de cache con TTL."""

    value: Dict[str, Any]
    created_at: float
    ttl_seconds: float

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds


class InMemoryPredictionCache:
    """Cache en memoria con LRU eviction y TTL.

    Attributes:
        _ttl_seconds: Tiempo de vida de cada entrada.
        _max_entries: Máximo de entradas (LRU eviction).
        _invalidation_threshold_pct: Cambio % para invalidar.
        _cache: OrderedDict para LRU.
        _hits: Contador de hits.
        _misses: Contador de misses.
        _lock: Lock para thread-safety.
    """

    def __init__(
        self,
        ttl_seconds: float = 60.0,
        max_entries: int = 1000,
        invalidation_threshold_pct: float = 0.1,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._invalidation_threshold_pct = invalidation_threshold_pct
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.Lock()

    def _make_key(
        self,
        sensor_id: int,
        recent_values: List[float],
        engine_name: str,
    ) -> str:
        """Genera key única para cache."""
        # Hash de últimos 5 valores + engine
        tail = recent_values[-5:] if len(recent_values) >= 5 else recent_values
        data = f"{sensor_id}:{[round(v, 4) for v in tail]}:{engine_name}"
        return f"pred:{hashlib.md5(data.encode()).hexdigest()}"

    def get(
        self,
        sensor_id: int,
        recent_values: List[float],
        engine_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Obtiene predicción cacheada si existe y no expiró.

        Args:
            sensor_id: ID del sensor.
            recent_values: Últimos valores del sensor.
            engine_name: Nombre del motor.

        Returns:
            Dict con predicción o ``None`` si miss/expirado.
        """
        key = self._make_key(sensor_id, recent_values, engine_name)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            # Mover al final (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(
        self,
        sensor_id: int,
        recent_values: List[float],
        engine_name: str,
        prediction_result: Dict[str, Any],
    ) -> None:
        """Cachea predicción.

        Args:
            sensor_id: ID del sensor.
            recent_values: Últimos valores.
            engine_name: Nombre del motor.
            prediction_result: Resultado a cachear.
        """
        key = self._make_key(sensor_id, recent_values, engine_name)

        with self._lock:
            # Eviction si excede max
            while len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)  # Eliminar más antiguo

            self._cache[key] = CacheEntry(
                value=prediction_result,
                created_at=time.monotonic(),
                ttl_seconds=self._ttl_seconds,
            )

    def should_invalidate(
        self,
        new_value: float,
        last_value: float,
    ) -> bool:
        """Decide si invalidar cache por cambio grande.

        Args:
            new_value: Nuevo valor del sensor.
            last_value: Último valor conocido.

        Returns:
            ``True`` si el cambio excede el threshold.
        """
        if abs(last_value) < 1e-9:
            return abs(new_value) > 0.1

        pct_change = abs(new_value - last_value) / abs(last_value)
        return pct_change > self._invalidation_threshold_pct

    def invalidate_sensor(self, sensor_id: int) -> int:
        """Invalida todas las entradas de un sensor.

        Args:
            sensor_id: ID del sensor.

        Returns:
            Número de entradas invalidadas.
        """
        prefix = f"pred:"
        count = 0

        with self._lock:
            keys_to_delete = [
                k for k in self._cache
                if f"{sensor_id}:" in k or k.startswith(prefix)
            ]
            for key in keys_to_delete:
                del self._cache[key]
                count += 1

        if count > 0:
            logger.debug(
                "cache_invalidated",
                extra={"sensor_id": sensor_id, "entries_removed": count},
            )

        return count

    def clear(self) -> None:
        """Limpia todo el cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Estadísticas del cache."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total,
                "hit_rate": round(hit_rate, 4),
                "entries": len(self._cache),
                "max_entries": self._max_entries,
            }
