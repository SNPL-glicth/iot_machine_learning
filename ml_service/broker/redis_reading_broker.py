"""Redis Streams implementation of ReadingBroker - Main entry point."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional

from .redis_broker_connection import RedisBrokerConnection
from .redis_broker_publisher import RedisBrokerPublisher
from .redis_broker_consumer import RedisBrokerConsumer
from ..reading_broker import Reading, ReadingBroker

logger = logging.getLogger(__name__)


class RedisReadingBroker(ReadingBroker):
    """Redis Streams implementation of ReadingBroker.
    
    Uses Redis Streams for durable, scalable message passing between
    Ingesta and ML services.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,  # DEPRECATED
        consumer_name: Optional[str] = None,
        max_stream_len: int = 10000,
    ) -> None:
        self._connection = RedisBrokerConnection(
            consumer_name=consumer_name or f"ml_{os.getpid()}",
            max_stream_len=max_stream_len,
        )
        self._publisher = RedisBrokerPublisher(self._connection)
        self._consumer = RedisBrokerConsumer(self._connection, handlers=[])
        self._handlers: List[Callable[[Reading], None]] = []
    
    def publish(self, reading: Reading) -> None:
        self._publisher.publish(reading)
    
    def subscribe(self, handler: Callable[[Reading], None]) -> None:
        self._handlers.append(handler)
        self._consumer._handlers = self._handlers  # Update consumer handlers
        
        if self._consumer._watchdog is None or not self._consumer._watchdog.is_healthy():
            self._consumer.start()
    
    def publish_ml_event(self, event: dict) -> None:
        self._connection.publish_ml_event(event)
    
    def stop(self) -> None:
        self._consumer.stop()
        self._connection.stop()
        logger.info("[REDIS_BROKER] Broker stopped")
    
    def is_healthy(self) -> bool:
        return self._consumer.is_healthy()
    
    def health_check(self) -> dict:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return {
            "connected": self._connection.is_connected,
            "consumer_running": self.is_healthy(),
            "handlers_count": len(self._handlers),
            "last_error": self._connection._last_error,
            "redis_url": redis_url.split("@")[-1],
            "consumer_name": self._connection.consumer_name,
            "stream_readings": RedisBrokerConnection.STREAM_READINGS,
            "stream_events": RedisBrokerConnection.STREAM_ML_EVENTS,
            "consumer_group": RedisBrokerConnection.CONSUMER_GROUP,
            "connection_source": "RedisConnectionManager (stream pool)",
            "circuit_breaker": self._publisher._circuit_breaker.get_metrics() if hasattr(self._publisher, '_circuit_breaker') else {},
            "local_queue_size": self._publisher.local_queue_size,
            "degraded_mode": self._publisher.is_degraded,
        }


__all__ = ["RedisReadingBroker"]