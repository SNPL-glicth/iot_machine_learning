"""RecentAnomalyTrackerAdapter — implementación Redis del Port.

Redis Schema:
    Key: anomaly_track:{series_id}
    Type: SortedSet (ZSET)
    Score: timestamp unix float
    Member: JSON {"anomaly_score": float, "regime": str, "is_anomaly": bool}
    TTL: 2 horas (ventana máxima)

Consecutive counter:
    Key: anomaly_consecutive:{series_id}
    Type: String (INCR/DEL)
    TTL: 2 horas
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from iot_machine_learning.domain.ports.recent_anomaly_tracker_port import (
    RecentAnomalyTrackerPort,
)
from iot_machine_learning.infrastructure.redis_keys import RedisKeys

logger = logging.getLogger(__name__)


class RecentAnomalyTrackerAdapter(RecentAnomalyTrackerPort):
    """Redis-backed anomaly tracker with in-memory fallback.
    
    Si Redis falla en cualquier operación:
    1. Loggear warning con contexto
    2. Usar in-memory fallback transparentemente
    3. Nunca lanzar excepción al caller
    """
    
    def __init__(
        self,
        redis_client: Any,
        ttl_seconds: float = 7200.0,
        fallback_tracker: Optional[RecentAnomalyTrackerPort] = None,
    ) -> None:
        """Inicializar adapter Redis.
        
        Args:
            redis_client: Cliente Redis (redis.Redis o compatible)
            ttl_seconds: TTL de keys (default 2 horas)
            fallback_tracker: Tracker alternativo si Redis falla
        """
        self._redis = redis_client
        self._ttl_seconds = int(ttl_seconds)
        self._fallback = fallback_tracker
        
        logger.info("redis_tracker_initialized", extra={"ttl": ttl_seconds})
    
    def _key_track(self, series_id: str) -> str:
        """Generar key para SortedSet de anomalías."""
        return RedisKeys.anomaly_track(series_id)

    def _key_consecutive(self, series_id: str) -> str:
        """Generar key para contador de consecutivas."""
        return RedisKeys.anomaly_consecutive(series_id)
    
    def _is_redis_available(self) -> bool:
        """Check rápido de disponibilidad."""
        return self._redis is not None
    
    def _exec_or_fallback(
        self,
        operation: str,
        series_id: str,
        redis_func,
        fallback_func,
    ) -> Any:
        """Ejecutar operación Redis o fallback si falla.
        
        Args:
            operation: Nombre de la operación para logging
            series_id: Serie afectada
            redis_func: Callable que ejecuta la operación Redis
            fallback_func: Callable que ejecuta el fallback
        
        Returns:
            Resultado de redis_func o fallback_func
        """
        if not self._is_redis_available():
            if self._fallback:
                return fallback_func()
            return None
        
        try:
            return redis_func()
        except Exception as e:
            logger.warning(
                "redis_op_failed_fallback",
                extra={
                    "operation": operation,
                    "series_id": series_id,
                    "error": str(e),
                },
            )
            if self._fallback:
                return fallback_func()
            # Valores por defecto seguros
            if operation in ("get_count", "get_consecutive"):
                return 0
            if operation == "get_rate":
                return 0.0
            return None
    
    def record_anomaly(
        self,
        series_id: str,
        anomaly_score: float,
        timestamp: Optional[float] = None,
        regime: str = "",
    ) -> None:
        """Registrar anomalía en Redis."""
        ts = timestamp if timestamp is not None else time.time()
        
        def _redis_op():
            key = self._key_track(series_id)
            member = json.dumps({
                "anomaly_score": anomaly_score,
                "regime": regime,
                "is_anomaly": True,
            })
            
            pipe = self._redis.pipeline()
            # Agregar a SortedSet
            pipe.zadd(key, {member: ts})
            # Eliminar entradas antiguas (más viejas que TTL)
            cutoff = ts - self._ttl_seconds
            pipe.zremrangebyscore(key, 0, cutoff)
            # Refrescar TTL de la key
            pipe.expire(key, self._ttl_seconds)
            # Incrementar contador de consecutivas
            pipe.incr(self._key_consecutive(series_id))
            pipe.expire(self._key_consecutive(series_id), self._ttl_seconds)
            pipe.execute()
        
        def _fallback_op():
            if self._fallback:
                self._fallback.record_anomaly(series_id, anomaly_score, ts, regime)
        
        self._exec_or_fallback("record_anomaly", series_id, _redis_op, _fallback_op)
    
    def record_normal(
        self,
        series_id: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Registrar predicción normal — resetea consecutivas."""
        ts = timestamp if timestamp is not None else time.time()
        
        def _redis_op():
            key = self._key_track(series_id)
            member = json.dumps({
                "anomaly_score": 0.0,
                "regime": "",
                "is_anomaly": False,
            })
            
            pipe = self._redis.pipeline()
            # Agregar entrada normal
            pipe.zadd(key, {member: ts})
            # Limpiar antiguas
            cutoff = ts - self._ttl_seconds
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.expire(key, self._ttl_seconds)
            # Resetear contador de consecutivas
            pipe.delete(self._key_consecutive(series_id))
            pipe.execute()
        
        def _fallback_op():
            if self._fallback:
                self._fallback.record_normal(series_id, ts)
        
        self._exec_or_fallback("record_normal", series_id, _redis_op, _fallback_op)
    
    def get_count_last_n_minutes(self, series_id: str, minutes: int) -> int:
        """Contar anomalías en ventana temporal."""
        def _redis_op() -> int:
            key = self._key_track(series_id)
            now = time.time()
            cutoff = now - (minutes * 60)
            
            # Obtener entradas en rango [cutoff, +inf]
            entries = self._redis.zrangebyscore(key, cutoff, "+inf")
            
            count = 0
            for entry in entries:
                try:
                    data = json.loads(entry.decode() if isinstance(entry, bytes) else entry)
                    if data.get("is_anomaly", False):
                        count += 1
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            return count
        
        def _fallback_op() -> int:
            if self._fallback:
                return self._fallback.get_count_last_n_minutes(series_id, minutes)
            return 0
        
        result = self._exec_or_fallback(
            "get_count", series_id, _redis_op, _fallback_op
        )
        return result if result is not None else 0
    
    def get_consecutive_count(self, series_id: str) -> int:
        """Obtener contador de consecutivas."""
        def _redis_op() -> int:
            key = self._key_consecutive(series_id)
            value = self._redis.get(key)
            if value is None:
                return 0
            try:
                return int(value.decode() if isinstance(value, bytes) else value)
            except (ValueError, AttributeError):
                return 0
        
        def _fallback_op() -> int:
            if self._fallback:
                return self._fallback.get_consecutive_count(series_id)
            return 0
        
        result = self._exec_or_fallback(
            "get_consecutive", series_id, _redis_op, _fallback_op
        )
        return result if result is not None else 0
    
    def get_anomaly_rate(self, series_id: str, window_minutes: int) -> float:
        """Calcular ratio anomalías / totales."""
        def _redis_op() -> float:
            key = self._key_track(series_id)
            now = time.time()
            cutoff = now - (window_minutes * 60)
            
            entries = self._redis.zrangebyscore(key, cutoff, "+inf")
            
            if not entries:
                return 0.0
            
            total = 0
            anomalies = 0
            
            for entry in entries:
                try:
                    data = json.loads(entry.decode() if isinstance(entry, bytes) else entry)
                    total += 1
                    if data.get("is_anomaly", False):
                        anomalies += 1
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            if total == 0:
                return 0.0
            
            return anomalies / total
        
        def _fallback_op() -> float:
            if self._fallback:
                return self._fallback.get_anomaly_rate(series_id, window_minutes)
            return 0.0
        
        result = self._exec_or_fallback(
            "get_rate", series_id, _redis_op, _fallback_op
        )
        return result if result is not None else 0.0
    
    def reset(self, series_id: Optional[str] = None) -> None:
        """Resetear historial."""
        def _redis_op():
            if series_id is None:
                # Borrar todas las keys de anomalías
                pattern_track = "anomaly_track:*"
                pattern_consecutive = "anomaly_consecutive:*"
                
                keys_track = self._redis.keys(pattern_track)
                keys_consecutive = self._redis.keys(pattern_consecutive)
                
                all_keys = (keys_track or []) + (keys_consecutive or [])
                if all_keys:
                    self._redis.delete(*all_keys)
            else:
                # Borrar keys específicas
                key_track = self._key_track(series_id)
                key_consecutive = self._key_consecutive(series_id)
                self._redis.delete(key_track, key_consecutive)
        
        def _fallback_op():
            if self._fallback:
                self._fallback.reset(series_id)
        
        self._exec_or_fallback("reset", series_id or "*", _redis_op, _fallback_op)
