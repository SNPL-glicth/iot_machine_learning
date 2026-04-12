"""Redis backend for ErrorHistoryManager.

Isolated Redis operations with transparent fallback.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, List, Optional, Set

logger = logging.getLogger(__name__)


class ErrorHistoryRedis:
    """Redis operations for error history with memory fallback."""
    
    def __init__(self, redis_client: Any, max_history: int = 50) -> None:
        self._redis = redis_client
        self._max_history = max_history
    
    def _get_key(self, series_id: str, engine_name: str) -> str:
        """Generate Redis key."""
        return f"error_history:{series_id}:{engine_name}"
    
    def record(
        self,
        series_id: str,
        engine_name: str,
        error: float,
        regime: str = "",
    ) -> bool:
        """Record error to Redis. Returns True on success."""
        try:
            key = self._get_key(series_id, engine_name)
            entry = json.dumps({
                "timestamp": time.time(),
                "error_value": error,
                "regime": regime,
            })
            
            pipe = self._redis.pipeline()
            pipe.lpush(key, entry)
            pipe.ltrim(key, 0, self._max_history - 1)
            pipe.execute()
            
            return True
        except Exception as e:
            logger.warning(
                "redis_record_failed",
                extra={"series_id": series_id, "engine": engine_name, "error": str(e)},
            )
            return False
    
    def get_errors(self, series_id: str, engine_name: str) -> Optional[List[float]]:
        """Get errors from Redis. Returns None on failure."""
        try:
            key = self._get_key(series_id, engine_name)
            entries = self._redis.lrange(key, 0, -1)
            
            errors = []
            for entry in entries:
                try:
                    data = json.loads(entry.decode() if isinstance(entry, bytes) else entry)
                    errors.append(data.get("error_value", 0.0))
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            return errors
        except Exception as e:
            logger.warning(
                "redis_get_failed",
                extra={"series_id": series_id, "engine": engine_name, "error": str(e)},
            )
            return None
    
    def get_all_engines(self, series_id: str) -> Optional[dict]:
        """Get all engines and their errors for a series. Returns None on failure."""
        try:
            pattern = self._get_key(series_id, "*")
            keys = self._redis.keys(pattern)
            
            result = {}
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) >= 3:
                    engine_name = parts[2]
                    errors = self.get_errors(series_id, engine_name)
                    if errors is not None:
                        result[engine_name] = errors
            
            return result
        except Exception as e:
            logger.warning(
                "redis_get_all_failed",
                extra={"series_id": series_id, "error": str(e)},
            )
            return None
    
    def reset(
        self,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> None:
        """Reset error history in Redis."""
        try:
            if series_id is None:
                keys = self._redis.keys("error_history:*")
                if keys:
                    self._redis.delete(*keys)
            elif engine_name is None:
                pattern = self._get_key(series_id, "*")
                keys = self._redis.keys(pattern)
                if keys:
                    self._redis.delete(*keys)
            else:
                key = self._get_key(series_id, engine_name)
                self._redis.delete(key)
        except Exception as e:
            logger.warning(
                "redis_reset_failed",
                extra={"series_id": series_id, "error": str(e)},
            )
    
    def get_all_series_ids(self) -> Set[str]:
        """Get all series IDs with error history."""
        series_ids = set()
        try:
            keys = self._redis.keys("error_history:*")
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) >= 2:
                    series_ids.add(parts[1])
        except Exception as e:
            logger.warning("redis_get_series_failed", extra={"error": str(e)})
        
        return series_ids
