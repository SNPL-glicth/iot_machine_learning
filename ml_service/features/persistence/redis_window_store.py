"""Redis-based window persistence for ML features.

FIX 2026-02-02: Implementa persistencia para evitar pérdida de contexto (ML-2).
FIX 2026-04-09:
- Uses RedisConnectionManager with async support
- Circuit breaker for resilience
- Staleness validation

Características:
- Persiste ventanas de sensores en Redis
- Recupera ventanas al reiniciar el servicio
- TTL configurable para limpieza automática
- Key prefix estandarizado: zenin:window:sensor:<sensor_id>
- Circuit breaker protection
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

from iot_machine_learning.infrastructure.persistence.redis import (
    RedisConnectionManager,
    get_redis_circuit_breaker,
)

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class PersistedWindow:
    """Ventana persistida en Redis."""
    sensor_id: int
    values: List[float]
    timestamps: List[float]
    last_updated: float


class RedisWindowStore:
    """Almacén de ventanas ML en Redis.
    
    Persiste las ventanas de sensores para recuperarlas después de un reinicio.
    """
    
    # MIGRATED: Standardized key prefix from ml:window: to zenin:window:sensor:
    KEY_PREFIX = "zenin:window:sensor:"
    DEFAULT_TTL_SECONDS = 3600  # 1 hora
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,  # Can be sync or async client
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_age_seconds: int = 120,  # Max age for window validity
    ):
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._max_age = max_age_seconds
        self._enabled = redis_client is not None
        
        # Circuit breaker for resilience
        self._circuit_breaker = get_redis_circuit_breaker("redis_window")
        
        if self._enabled:
            logger.info(
                "ML WindowStore initialized (ttl=%ds, max_age=%ds)",
                ttl_seconds,
                max_age_seconds
            )
        else:
            logger.warning("ML WindowStore disabled (no Redis client)")
    
    def _key(self, sensor_id: int) -> str:
        """Genera la clave Redis para un sensor."""
        return f"{self.KEY_PREFIX}{sensor_id}"
    
    def _save_sync(self, sensor_id: int, values: List[float], timestamps: List[float]) -> bool:
        """Synchronous save implementation."""
        window = PersistedWindow(
            sensor_id=sensor_id,
            values=values[-100:],  # Limitar a últimos 100 valores
            timestamps=timestamps[-100:],
            last_updated=time.time(),
        )
        
        data = json.dumps(asdict(window))
        self._redis.setex(self._key(sensor_id), self._ttl, data)
        
        logger.debug("Saved window: sensor_id=%d values=%d", sensor_id, len(values))
        return True
    
    def save(self, sensor_id: int, values: List[float], timestamps: List[float]) -> bool:
        """Persiste una ventana en Redis con circuit breaker protection.
        
        Args:
            sensor_id: ID del sensor
            values: Lista de valores en la ventana
            timestamps: Lista de timestamps correspondientes
            
        Returns:
            True si se guardó correctamente, False si falló
        """
        if not self._enabled or not self._redis:
            return False
        
        def _fallback():
            logger.debug("Window save skipped (circuit open): sensor_id=%d", sensor_id)
            return False
        
        try:
            return self._circuit_breaker.call(
                lambda: self._save_sync(sensor_id, values, timestamps),
                _fallback
            )
        except Exception as e:
            logger.warning(
                "Failed to save window: sensor_id=%d error=%s",
                sensor_id,
                e
            )
            return False
    
    async def save_async(
        self,
        sensor_id: int,
        values: List[float],
        timestamps: List[float]
    ) -> bool:
        """Async version of save for use in async contexts."""
        if not self._enabled or not self._redis:
            return False
        
        def _fallback():
            logger.debug("Window save skipped (circuit open): sensor_id=%d", sensor_id)
            return False
        
        try:
            window = PersistedWindow(
                sensor_id=sensor_id,
                values=values[-100:],
                timestamps=timestamps[-100:],
                last_updated=time.time(),
            )
            
            data = json.dumps(asdict(window))
            
            # Use async client
            async_client = await RedisConnectionManager.get_async_client()
            await async_client.setex(self._key(sensor_id), self._ttl, data.encode())
            
            logger.debug("Saved window (async): sensor_id=%d", sensor_id)
            return True
            
        except Exception as e:
            logger.warning(
                "Failed to save window (async): sensor_id=%d error=%s",
                sensor_id,
                e
            )
            return False
    
    def _load_sync(self, sensor_id: int) -> Optional[PersistedWindow]:
        """Synchronous load implementation with staleness check."""
        data = self._redis.get(self._key(sensor_id))
        if data is None:
            return None
        
        parsed = json.loads(data)
        window = PersistedWindow(**parsed)
        
        # Staleness validation
        age = time.time() - window.last_updated
        if age > self._max_age:
            logger.warning(
                "Stale window rejected: sensor_id=%d age=%.1fs (max=%ds)",
                sensor_id,
                age,
                self._max_age
            )
            # Delete stale data
            self._redis.delete(self._key(sensor_id))
            return None
        
        logger.debug(
            "Loaded window: sensor_id=%d values=%d age=%.1fs",
            sensor_id,
            len(window.values),
            age
        )
        return window
    
    def load(
        self,
        sensor_id: int,
        max_age_seconds: Optional[int] = None
    ) -> Optional[PersistedWindow]:
        """Carga una ventana desde Redis con validación de frescura.
        
        Args:
            sensor_id: ID del sensor
            max_age_seconds: Override default max age for this load
            
        Returns:
            PersistedWindow si existe y es fresca, None si no
        """
        if not self._enabled or not self._redis:
            return None
        
        # Temporarily override max_age if provided
        original_max_age = self._max_age
        if max_age_seconds is not None:
            self._max_age = max_age_seconds
        
        def _fallback():
            logger.debug("Window load skipped (circuit open): sensor_id=%d", sensor_id)
            return None
        
        try:
            result = self._circuit_breaker.call(
                lambda: self._load_sync(sensor_id),
                _fallback
            )
            return result
        except Exception as e:
            logger.warning(
                "Failed to load window: sensor_id=%d error=%s",
                sensor_id,
                e
            )
            return None
        finally:
            self._max_age = original_max_age
    
    async def load_async(
        self,
        sensor_id: int,
        max_age_seconds: Optional[int] = None
    ) -> Optional[PersistedWindow]:
        """Async version of load for use in async contexts."""
        if not self._enabled or not self._redis:
            return None
        
        max_age = max_age_seconds or self._max_age
        
        try:
            async_client = await RedisConnectionManager.get_async_client()
            data = await async_client.get(self._key(sensor_id))
            
            if data is None:
                return None
            
            parsed = json.loads(data.decode())
            window = PersistedWindow(**parsed)
            
            # Staleness validation
            age = time.time() - window.last_updated
            if age > max_age:
                logger.warning(
                    "Stale window rejected (async): sensor_id=%d age=%.1fs",
                    sensor_id,
                    age
                )
                await async_client.delete(self._key(sensor_id))
                return None
            
            logger.debug("Loaded window (async): sensor_id=%d", sensor_id)
            return window
            
        except Exception as e:
            logger.warning(
                "Failed to load window (async): sensor_id=%d error=%s",
                sensor_id,
                e
            )
            return None
    
    def delete(self, sensor_id: int) -> bool:
        """Elimina una ventana de Redis."""
        if not self._enabled or not self._redis:
            return False
        
        try:
            self._redis.delete(self._key(sensor_id))
            return True
        except Exception:
            return False
    
    def get_all_sensor_ids(self) -> List[int]:
        """Obtiene todos los sensor_ids con ventanas persistidas."""
        if not self._enabled or not self._redis:
            return []
        
        try:
            pattern = f"{self.KEY_PREFIX}*"
            keys = self._redis.keys(pattern)
            
            sensor_ids = []
            prefix_len = len(self.KEY_PREFIX)
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                sensor_id_str = key_str[prefix_len:]
                try:
                    sensor_ids.append(int(sensor_id_str))
                except ValueError:
                    pass
            
            return sensor_ids
            
        except Exception as e:
            logger.warning("Failed to get sensor IDs: %s", e)
            return []
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    @property
    def stats(self) -> dict:
        """Estadísticas del store."""
        return {
            "enabled": self._enabled,
            "ttl_seconds": self._ttl,
            "max_age_seconds": self._max_age,
            "sensor_count": len(self.get_all_sensor_ids()) if self._enabled else 0,
            "circuit_breaker": self._circuit_breaker.get_metrics(),
        }


# Singleton
_store_instance: Optional[RedisWindowStore] = None


def get_window_store() -> RedisWindowStore:
    """Obtiene el store singleton.
    
    Configuración via variables de entorno:
    - REDIS_URL: URL de Redis (default: redis://localhost:6379/0)
    - ML_WINDOW_TTL_SECONDS: TTL de ventanas (default: 3600)
    - ML_WINDOW_MAX_AGE_SECONDS: Max age for window validity (default: 120)
    
    MIGRATED: Now uses RedisConnectionManager with circuit breaker.
    """
    global _store_instance
    
    if _store_instance is not None:
        return _store_instance
    
    if not REDIS_AVAILABLE:
        logger.warning("ML WindowStore: redis package not installed")
        _store_instance = RedisWindowStore(redis_client=None)
        return _store_instance
    
    import os
    ttl = int(os.getenv("ML_WINDOW_TTL_SECONDS", "3600"))
    max_age = int(os.getenv("ML_WINDOW_MAX_AGE_SECONDS", "120"))
    
    try:
        # Use centralized connection manager with circuit breaker protection
        client = RedisConnectionManager.get_sync_client()
        logger.info(
            "ML WindowStore connected (ttl=%ds, max_age=%ds)",
            ttl,
            max_age
        )
        _store_instance = RedisWindowStore(
            redis_client=client,
            ttl_seconds=ttl,
            max_age_seconds=max_age
        )
        
    except Exception as e:
        logger.warning("ML WindowStore connection failed: %s", e)
        _store_instance = RedisWindowStore(redis_client=None)
    
    return _store_instance


def reset_window_store() -> None:
    """Resetea el singleton (para tests)."""
    global _store_instance
    _store_instance = None
