"""Redis-backed sliding window store — FASE 1 CRÍTICO.

Migrates sliding windows from RAM to Redis for persistence across restarts.

Fixes LEAK-1: Windows survive service restarts.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Reading:
    """Single sensor reading."""

    sensor_id: int
    value: float
    timestamp: float


class RedisSlidingWindowStore:
    """Redis-backed sliding window store with TTL.

    Stores sliding windows in Redis lists with automatic expiration.

    Attributes:
        _redis_client: Redis client
        _window_size: Max readings per window
        _ttl_seconds: TTL for inactive windows
        _key_prefix: Redis key prefix
    """

    def __init__(
        self,
        redis_client,
        window_size: int = 20,
        ttl_seconds: int = 3600,
        key_prefix: str = "sliding_window:",
    ):
        """Initialize Redis sliding window store.

        Args:
            redis_client: Redis client instance
            window_size: Max readings per window
            ttl_seconds: TTL for inactive windows (default 1 hour)
            key_prefix: Redis key prefix
        """
        self._redis = redis_client
        self._window_size = window_size
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix

    def append(self, reading: Reading) -> None:
        """Append reading to window.

        Args:
            reading: Reading to append
        """
        try:
            key = self._make_key(reading.sensor_id)

            # Serialize reading
            data = json.dumps({
                "sensor_id": reading.sensor_id,
                "value": reading.value,
                "timestamp": reading.timestamp,
            })

            # Push to list (right side = newest)
            self._redis.rpush(key, data)

            # Trim to window size
            self._redis.ltrim(key, -self._window_size, -1)

            # Set TTL
            self._redis.expire(key, self._ttl_seconds)

        except Exception as exc:
            logger.error(
                "redis_window_append_failed",
                extra={
                    "sensor_id": reading.sensor_id,
                    "error": str(exc),
                },
            )

    def get_window(self, sensor_id: int) -> Deque[Reading]:
        """Get window for sensor.

        Args:
            sensor_id: Sensor ID

        Returns:
            Deque of readings (oldest first)
        """
        try:
            key = self._make_key(sensor_id)

            # Get all readings from list
            data_list = self._redis.lrange(key, 0, -1)

            # Deserialize
            readings = deque()
            for data in data_list:
                try:
                    obj = json.loads(data)
                    readings.append(Reading(
                        sensor_id=obj["sensor_id"],
                        value=obj["value"],
                        timestamp=obj["timestamp"],
                    ))
                except Exception:
                    continue

            return readings

        except Exception as exc:
            logger.error(
                "redis_window_get_failed",
                extra={
                    "sensor_id": sensor_id,
                    "error": str(exc),
                },
            )
            return deque()

    def clear(self, sensor_id: int) -> None:
        """Clear window for sensor.

        Args:
            sensor_id: Sensor ID
        """
        try:
            key = self._make_key(sensor_id)
            self._redis.delete(key)

        except Exception as exc:
            logger.error(
                "redis_window_clear_failed",
                extra={
                    "sensor_id": sensor_id,
                    "error": str(exc),
                },
            )

    def sensor_ids(self) -> List[int]:
        """Get all sensor IDs with windows.

        Returns:
            List of sensor IDs
        """
        try:
            pattern = f"{self._key_prefix}*"
            keys = self._redis.keys(pattern)

            sensor_ids = []
            for key in keys:
                try:
                    # Extract sensor_id from key
                    sensor_id_str = key.decode().replace(self._key_prefix, "")
                    sensor_ids.append(int(sensor_id_str))
                except Exception:
                    continue

            return sensor_ids

        except Exception as exc:
            logger.error(
                "redis_window_sensor_ids_failed",
                extra={"error": str(exc)},
            )
            return []

    def get_metrics(self) -> dict:
        """Get store metrics.

        Returns:
            Dict with metrics
        """
        try:
            sensor_ids = self.sensor_ids()
            return {
                "active_sensors": len(sensor_ids),
                "backend": "redis",
                "window_size": self._window_size,
                "ttl_seconds": self._ttl_seconds,
            }

        except Exception:
            return {
                "active_sensors": 0,
                "backend": "redis",
                "error": True,
            }

    def _make_key(self, sensor_id: int) -> str:
        """Make Redis key for sensor.

        Args:
            sensor_id: Sensor ID

        Returns:
            Redis key
        """
        return f"{self._key_prefix}{sensor_id}"
