"""Redis store mixin for AlertSuppressor.

Handles all Redis I/O: reading last alert, saving alerts, and
incrementing suppression counters.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from ....infrastructure.redis.redis_keys import RedisKeys

logger = logging.getLogger(__name__)


class _AlertStoreMixin:
    """Mixin that provides Redis-backed alert storage."""

    def _get_redis_key(self, series_id: str) -> str:
        """Generar key para última alerta."""
        return RedisKeys.last_alert(series_id)

    def _get_last_alert(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Recuperar última alerta desde Redis."""
        redis = getattr(self, "_redis", None)
        if redis is None:
            return None

        try:
            key = self._get_redis_key(series_id)
            data = redis.get(key)
            if data is None:
                return None

            json_str = data.decode() if isinstance(data, bytes) else data
            return json.loads(json_str)
        except (ConnectionError, TimeoutError, json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.warning(
                "redis_get_last_alert_failed",
                extra={
                    "series_id": series_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            return None

    def _save_alert(
        self,
        series_id: str,
        action: str,
        priority: int,
        severity: str,
    ) -> None:
        """Guardar alerta emitida en Redis."""
        redis = getattr(self, "_redis", None)
        if redis is None:
            return

        try:
            key = self._get_redis_key(series_id)
            value = json.dumps({
                "action": action,
                "priority": priority,
                "timestamp": time.time(),
                "severity": severity,
            })
            redis.setex(key, self._get_key_ttl(), value)
        except (ConnectionError, TimeoutError, TypeError, AttributeError) as e:
            logger.warning(
                "redis_save_alert_failed",
                extra={
                    "series_id": series_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )

    def _increment_suppressed(self, series_id: str) -> int:
        """Incrementar contador de suprimidas. Retorna nuevo valor."""
        redis = getattr(self, "_redis", None)
        if redis is None:
            return 0

        try:
            key = RedisKeys.suppressed(series_id)
            count = redis.incr(key)
            redis.expire(key, self._get_key_ttl())
            return int(count) if count else 0
        except (ConnectionError, TimeoutError, TypeError, AttributeError) as e:
            logger.warning(
                "redis_increment_suppressed_failed",
                extra={
                    "series_id": series_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            return 0

    def get_suppressed_count(self, series_id: str) -> int:
        """Obtener contador de alertas suprimidas para una serie."""
        redis = getattr(self, "_redis", None)
        if redis is None:
            return 0

        try:
            key = f"suppressed:{series_id}"
            value = redis.get(key)
            if value is None:
                return 0
            return int(value.decode() if isinstance(value, bytes) else value)
        except (ConnectionError, TimeoutError, TypeError, ValueError, AttributeError) as e:
            logger.warning(
                "redis_get_suppressed_count_failed",
                extra={
                    "series_id": series_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            return 0
