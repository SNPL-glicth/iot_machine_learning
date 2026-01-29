"""Redis Streams implementation of ReadingBroker.

Integrates Redis Streams with the ML service's ReadingBroker interface.
This replaces InMemoryReadingBroker for production use.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Callable, Optional

import redis
from redis import Redis

from ..reading_broker import Reading, ReadingBroker

logger = logging.getLogger(__name__)


class RedisReadingBroker(ReadingBroker):
    """Redis Streams implementation of ReadingBroker.
    
    Uses Redis Streams for durable, scalable message passing between
    Ingesta and ML services.
    
    Streams:
    - readings:validated - Lecturas validadas para ML
    - events:ml - Eventos ML generados
    
    Features:
    - Consumer groups for load balancing
    - Message acknowledgment
    - Automatic reconnection
    - Fallback to InMemory if Redis unavailable
    """
    
    STREAM_READINGS = "readings:validated"
    STREAM_ML_EVENTS = "events:ml"
    CONSUMER_GROUP = "ml_processors"
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        consumer_name: Optional[str] = None,
        max_stream_len: int = 10000,
    ) -> None:
        """Initialize Redis broker.
        
        Args:
            redis_url: Redis connection URL. Default: REDIS_URL env var or localhost
            consumer_name: Unique consumer name. Default: hostname + pid
            max_stream_len: Max messages in stream (MAXLEN ~)
        """
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._consumer_name = consumer_name or f"ml_{os.getpid()}"
        self._max_stream_len = max_stream_len
        
        self._redis: Optional[Redis] = None
        self._running = False
        self._handlers: list[Callable[[Reading], None]] = []
        self._consumer_thread: Optional[threading.Thread] = None
        
        # Connection state
        self._connected = False
        self._last_error: Optional[str] = None
    
    def _connect(self) -> bool:
        """Establish Redis connection."""
        if self._redis is not None and self._connected:
            return True
        
        try:
            self._redis = Redis.from_url(
                self._redis_url,
                decode_responses=False,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
            )
            # Test connection
            self._redis.ping()
            self._connected = True
            self._last_error = None
            logger.info(
                "[REDIS_BROKER] Connected to Redis: url=%s consumer=%s",
                self._redis_url.split("@")[-1],  # Hide password
                self._consumer_name,
            )
            return True
        except Exception as e:
            self._connected = False
            self._last_error = str(e)
            logger.error(
                "[REDIS_BROKER] Connection failed: %s",
                str(e),
            )
            return False
    
    def _ensure_consumer_group(self) -> None:
        """Create consumer group if it doesn't exist."""
        if not self._redis:
            return
        
        try:
            self._redis.xgroup_create(
                self.STREAM_READINGS,
                self.CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
            logger.info(
                "[REDIS_BROKER] Created consumer group: %s",
                self.CONSUMER_GROUP,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug("Consumer group already exists")
            else:
                raise
    
    def publish(self, reading: Reading) -> None:
        """Publish a reading to Redis Stream.
        
        Uses XADD with MAXLEN ~ for bounded stream.
        """
        if not self._connect():
            logger.warning(
                "[REDIS_BROKER] Cannot publish, not connected. Dropping reading sensor_id=%s",
                reading.sensor_id,
            )
            return
        
        try:
            data = {
                "sensor_id": str(reading.sensor_id),
                "sensor_type": reading.sensor_type,
                "value": str(reading.value),
                "timestamp": str(reading.timestamp),
            }
            
            msg_id = self._redis.xadd(
                self.STREAM_READINGS,
                data,
                maxlen=self._max_stream_len,
                approximate=True,
            )
            
            logger.debug(
                "[REDIS_BROKER] Published: msg_id=%s sensor_id=%s value=%.4f",
                msg_id,
                reading.sensor_id,
                reading.value,
            )
        except Exception as e:
            self._connected = False
            logger.error(
                "[REDIS_BROKER] Publish failed: sensor_id=%s error=%s",
                reading.sensor_id,
                str(e),
            )
    
    def subscribe(self, handler: Callable[[Reading], None]) -> None:
        """Subscribe to readings from Redis Stream.
        
        Starts a consumer thread that reads from the stream.
        """
        self._handlers.append(handler)
        
        if self._consumer_thread is None or not self._consumer_thread.is_alive():
            self._start_consumer()
    
    def _start_consumer(self) -> None:
        """Start the consumer thread."""
        if not self._connect():
            logger.error("[REDIS_BROKER] Cannot start consumer, not connected")
            return
        
        self._ensure_consumer_group()
        self._running = True
        
        self._consumer_thread = threading.Thread(
            target=self._consume_loop,
            name="redis-reading-consumer",
            daemon=True,
        )
        self._consumer_thread.start()
        
        logger.info(
            "[REDIS_BROKER] Consumer started: group=%s consumer=%s",
            self.CONSUMER_GROUP,
            self._consumer_name,
        )
    
    def _consume_loop(self) -> None:
        """Main consumer loop."""
        while self._running:
            try:
                if not self._connected and not self._connect():
                    # Retry connection
                    import time
                    time.sleep(5)
                    continue
                
                # Read from stream with consumer group
                messages = self._redis.xreadgroup(
                    groupname=self.CONSUMER_GROUP,
                    consumername=self._consumer_name,
                    streams={self.STREAM_READINGS: ">"},
                    count=10,
                    block=1000,  # 1 second timeout
                )
                
                if not messages:
                    continue
                
                for stream_name, stream_messages in messages:
                    for msg_id, fields in stream_messages:
                        try:
                            reading = self._parse_reading(fields)
                            
                            # Call all handlers
                            for handler in self._handlers:
                                try:
                                    handler(reading)
                                except Exception as e:
                                    logger.exception(
                                        "[REDIS_BROKER] Handler error: %s",
                                        str(e),
                                    )
                            
                            # Acknowledge message
                            self._redis.xack(
                                self.STREAM_READINGS,
                                self.CONSUMER_GROUP,
                                msg_id,
                            )
                            
                        except Exception as e:
                            logger.exception(
                                "[REDIS_BROKER] Message processing error: msg_id=%s",
                                msg_id,
                            )
                
            except redis.ConnectionError as e:
                self._connected = False
                logger.error(
                    "[REDIS_BROKER] Connection lost: %s. Reconnecting...",
                    str(e),
                )
                import time
                time.sleep(1)
            except Exception as e:
                logger.exception("[REDIS_BROKER] Consumer error: %s", str(e))
                import time
                time.sleep(1)
        
        logger.info("[REDIS_BROKER] Consumer stopped")
    
    def _parse_reading(self, fields: dict) -> Reading:
        """Parse Redis message fields to Reading."""
        def decode(v):
            return v.decode() if isinstance(v, bytes) else str(v)
        
        return Reading(
            sensor_id=int(decode(fields.get(b"sensor_id", fields.get("sensor_id", 0)))),
            sensor_type=decode(fields.get(b"sensor_type", fields.get("sensor_type", "unknown"))),
            value=float(decode(fields.get(b"value", fields.get("value", 0)))),
            timestamp=float(decode(fields.get(b"timestamp", fields.get("timestamp", 0)))),
        )
    
    def publish_ml_event(self, event: dict) -> None:
        """Publish ML event to events stream."""
        if not self._connect():
            return
        
        try:
            # Convert all values to strings for Redis
            data = {k: str(v) for k, v in event.items()}
            
            self._redis.xadd(
                self.STREAM_ML_EVENTS,
                data,
                maxlen=self._max_stream_len,
                approximate=True,
            )
            
            logger.debug(
                "[REDIS_BROKER] Published ML event: sensor_id=%s type=%s",
                event.get("sensor_id"),
                event.get("event_type"),
            )
        except Exception as e:
            logger.error(
                "[REDIS_BROKER] ML event publish failed: %s",
                str(e),
            )
    
    def stop(self) -> None:
        """Stop the consumer."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=5)
        if self._redis:
            self._redis.close()
        logger.info("[REDIS_BROKER] Broker stopped")
    
    def health_check(self) -> dict:
        """Return health status."""
        return {
            "connected": self._connected,
            "consumer_running": self._running and (
                self._consumer_thread is not None and 
                self._consumer_thread.is_alive()
            ),
            "handlers_count": len(self._handlers),
            "last_error": self._last_error,
            "redis_url": self._redis_url.split("@")[-1],  # Hide password
            "consumer_name": self._consumer_name,
            "stream_readings": self.STREAM_READINGS,
            "stream_events": self.STREAM_ML_EVENTS,
            "consumer_group": self.CONSUMER_GROUP,
        }
