"""Redis broker connection management."""

from __future__ import annotations

import logging
import os
from typing import Optional

import redis
from redis import Redis

from iot_machine_learning.infrastructure.persistence.redis import (
    RedisConnectionManager,
    get_redis_circuit_breaker,
)

logger = logging.getLogger(__name__)


class RedisBrokerConnection:
    """Manages Redis connection for broker with circuit breaker."""
    
    STREAM_READINGS = "readings:validated"
    STREAM_ML_EVENTS = "events:ml"
    CONSUMER_GROUP = "ml_processors"
    
    def __init__(
        self,
        consumer_name: Optional[str] = None,
        max_stream_len: int = 10000,
    ):
        self._consumer_name = consumer_name or f"ml_{os.getpid()}"
        self._max_stream_len = max_stream_len
        self._redis: Optional[Redis] = None
        self._connected = False
        self._last_error: Optional[str] = None
        self._circuit_breaker = get_redis_circuit_breaker("redis_broker")
    
    def _connect(self) -> bool:
        if self._redis is not None and self._connected:
            return True
        
        def _do_connect():
            self._redis = RedisConnectionManager.get_stream_client()
            self._redis.ping()
            return True
        
        def _fallback():
            logger.debug("[REDIS_BROKER] Operating in degraded mode (no Redis). Readings queued locally.")
            return False
        
        try:
            connected = self._circuit_breaker.call(_do_connect, _fallback)
            self._connected = connected
            
            if connected:
                self._last_error = None
                logger.info("[REDIS_BROKER] Connected via stream pool consumer=%s", self._consumer_name)
            return connected
        except Exception as e:
            self._connected = False
            self._last_error = str(e)
            logger.error("[REDIS_BROKER] Connection failed: %s", str(e))
            return False
    
    def ensure_consumer_group(self) -> None:
        if not self._redis:
            return
        try:
            self._redis.xgroup_create(
                self.STREAM_READINGS,
                self.CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
            logger.info("[REDIS_BROKER] Created consumer group: %s", self.CONSUMER_GROUP)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
    
    def publish_to_redis(self, reading) -> Optional[str]:
        data = {
            "sensor_id": str(reading.sensor_id),
            "sensor_type": reading.sensor_type,
            "value": str(reading.value),
            "timestamp": str(reading.timestamp),
        }
        return self._redis.xadd(
            self.STREAM_READINGS,
            data,
            maxlen=self._max_stream_len,
            approximate=True,
        )
    
    def publish_ml_event(self, event: dict) -> None:
        if not self._connect():
            return
        data = {k: str(v) for k, v in event.items()}
        self._redis.xadd(
            self.STREAM_ML_EVENTS,
            data,
            maxlen=self._max_stream_len,
            approximate=True,
        )
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def redis_client(self) -> Optional[Redis]:
        return self._redis
    
    @property
    def consumer_name(self) -> str:
        return self._consumer_name
    
    def stop(self) -> None:
        if self._redis:
            self._redis.close()