"""Redis-based window persistence for ML features.

FIX 2026-02-02: Implementa persistencia para evitar pérdida de contexto (ML-2).

Características:
- Persiste ventanas de sensores en Redis
- Recupera ventanas al reiniciar el servicio
- TTL configurable para limpieza automática
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

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
    
    KEY_PREFIX = "ml:window:"
    DEFAULT_TTL_SECONDS = 3600  # 1 hora
    
    def __init__(
        self,
        redis_client: Optional["redis.Redis"] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._enabled = redis_client is not None
        
        if self._enabled:
            logger.info("ML WindowStore initialized with Redis (ttl=%ds)", ttl_seconds)
        else:
            logger.warning("ML WindowStore disabled (no Redis client)")
    
    def _key(self, sensor_id: int) -> str:
        """Genera la clave Redis para un sensor."""
        return f"{self.KEY_PREFIX}{sensor_id}"
    
    def save(self, sensor_id: int, values: List[float], timestamps: List[float]) -> bool:
        """Persiste una ventana en Redis.
        
        Args:
            sensor_id: ID del sensor
            values: Lista de valores en la ventana
            timestamps: Lista de timestamps correspondientes
            
        Returns:
            True si se guardó correctamente, False si falló
        """
        if not self._enabled or not self._redis:
            return False
        
        try:
            import time
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
            
        except Exception as e:
            logger.warning("Failed to save window: sensor_id=%d error=%s", sensor_id, e)
            return False
    
    def load(self, sensor_id: int) -> Optional[PersistedWindow]:
        """Carga una ventana desde Redis.
        
        Args:
            sensor_id: ID del sensor
            
        Returns:
            PersistedWindow si existe, None si no
        """
        if not self._enabled or not self._redis:
            return None
        
        try:
            data = self._redis.get(self._key(sensor_id))
            if data is None:
                return None
            
            parsed = json.loads(data)
            window = PersistedWindow(**parsed)
            
            logger.debug("Loaded window: sensor_id=%d values=%d", sensor_id, len(window.values))
            return window
            
        except Exception as e:
            logger.warning("Failed to load window: sensor_id=%d error=%s", sensor_id, e)
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
            "sensor_count": len(self.get_all_sensor_ids()) if self._enabled else 0,
        }


# Singleton
_store_instance: Optional[RedisWindowStore] = None


def get_window_store() -> RedisWindowStore:
    """Obtiene el store singleton.
    
    Configuración via variables de entorno:
    - ML_REDIS_URL: URL de Redis (default: redis://localhost:6379/0)
    - ML_WINDOW_TTL_SECONDS: TTL de ventanas (default: 3600)
    """
    global _store_instance
    
    if _store_instance is not None:
        return _store_instance
    
    if not REDIS_AVAILABLE:
        logger.warning("ML WindowStore: redis package not installed")
        _store_instance = RedisWindowStore(redis_client=None)
        return _store_instance
    
    redis_url = os.getenv("ML_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    ttl = int(os.getenv("ML_WINDOW_TTL_SECONDS", "3600"))
    
    try:
        client = redis.from_url(redis_url, decode_responses=False)
        client.ping()
        logger.info("ML WindowStore connected to Redis: %s", redis_url.split("@")[-1])
        _store_instance = RedisWindowStore(redis_client=client, ttl_seconds=ttl)
        
    except Exception as e:
        logger.warning("ML WindowStore Redis connection failed: %s", e)
        _store_instance = RedisWindowStore(redis_client=None)
    
    return _store_instance


def reset_window_store() -> None:
    """Resetea el singleton (para tests)."""
    global _store_instance
    _store_instance = None
