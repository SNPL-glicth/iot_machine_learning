"""Redis adapter implementing AlertStateRepositoryPort."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from iot_machine_learning.domain.ports.alert_state_repository_port import AlertStateRepositoryPort
from iot_machine_learning.infrastructure.redis.redis_keys import RedisKeys

logger = logging.getLogger(__name__)


class RedisAlertStateAdapter(AlertStateRepositoryPort):
    """Redis-backed implementation of AlertStateRepositoryPort."""

    def __init__(self, redis_client: Any, ttl_seconds: int = 3600, keys: Optional[RedisKeys] = None) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._keys = keys or RedisKeys()

    def _get_redis_key(self, series_id: str) -> str:
        return self._keys.last_alert(series_id)

    def get_last_alert(self, series_id: str) -> Optional[Dict[str, Any]]:
        if self._redis is None:
            return None

        try:
            key = self._get_redis_key(series_id)
            data = self._redis.get(key)
            if data is None:
                return None

            json_str = data.decode() if isinstance(data, bytes) else data
            parsed = json.loads(json_str)
            return parsed if isinstance(parsed, dict) else None
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

    def save_alert(
        self,
        series_id: str,
        action: str,
        priority: int,
        severity: str,
    ) -> None:
        if self._redis is None:
            return

        try:
            key = self._get_redis_key(series_id)
            value = json.dumps({
                "action": action,
                "priority": priority,
                "timestamp": time.time(),
                "severity": severity,
            })
            self._redis.setex(key, self._ttl_seconds, value)
        except (ConnectionError, TimeoutError, TypeError, AttributeError) as e:
            logger.warning(
                "redis_save_alert_failed",
                extra={
                    "series_id": series_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )

    def increment_suppressed(self, series_id: str) -> int:
        if self._redis is None:
            return 0

        try:
            key = self._keys.suppressed(series_id)
            count = self._redis.incr(key)
            self._redis.expire(key, self._ttl_seconds)
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
        if self._redis is None:
            return 0

        try:
            key = f"suppressed:{series_id}"
            value = self._redis.get(key)
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
