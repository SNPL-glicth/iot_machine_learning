"""Redis broker publisher with circuit breaker and local fallback."""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Optional

from iot_machine_learning.infrastructure.persistence.redis import get_redis_circuit_breaker

from .redis_broker_connection import RedisBrokerConnection
from ..reading_broker import Reading, ReadingBroker

logger = logging.getLogger(__name__)


class RedisBrokerPublisher:
    """Handles publishing readings with circuit breaker and local fallback."""
    
    def __init__(self, connection: RedisBrokerConnection):
        self._connection = connection
        self._local_queue: deque[Reading] = deque(maxlen=10000)
        self._circuit_breaker = get_redis_circuit_breaker("redis_broker_publisher")
    
    def publish(self, reading: Reading) -> None:
        if not self._connection._connect():
            self._queue_locally(reading, "Redis unavailable")
            return
        
        def _do_publish():
            return self._connection.publish_to_redis(reading)
        
        def _fallback():
            self._queue_locally(reading, "Circuit open")
            return None
        
        try:
            msg_id = self._circuit_breaker.call(_do_publish, _fallback)
            if msg_id:
                logger.debug(
                    "[REDIS_BROKER] Published: msg_id=%s sensor_id=%s value=%.4f",
                    msg_id, reading.sensor_id, reading.value
                )
        except Exception as e:
            self._connection._connected = False
            self._queue_locally(reading, f"Publish failed: {e}")
    
    def _queue_locally(self, reading: Reading, reason: str) -> None:
        self._local_queue.append(reading)
        logger.warning("[REDIS_BROKER] %s. Queued locally. Queue size: %d", reason, len(self._local_queue))
    
    def flush_local_queue(self) -> None:
        while self._connection._local_queue and self._connection.is_connected:
            try:
                reading = self._connection._local_queue.popleft()
                self._connection.publish_to_redis(reading)
            except Exception as e:
                logger.error("[REDIS_BROKER] Failed to flush queued reading: %s", e)
                break
    
    @property
    def local_queue_size(self) -> int:
        return len(self._local_queue)
    
    @property
    def is_degraded(self) -> bool:
        return not self._connection.is_connected and len(self._local_queue) > 0