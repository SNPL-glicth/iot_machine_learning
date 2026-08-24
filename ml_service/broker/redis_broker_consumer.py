"""Redis broker consumer loop with watchdog."""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

import redis

from iot_machine_learning.ml_service.utils.watchdog_thread import WatchdogThread
from .redis_broker_connection import RedisBrokerConnection
from ..reading_broker import Reading, ReadingBroker

logger = logging.getLogger(__name__)


class RedisBrokerConsumer:
    """Consumer loop for Redis Streams with watchdog supervision."""
    
    def __init__(self, connection: RedisBrokerConnection, handlers: List[Callable[[Reading], None]]):
        self._connection = connection
        self._handlers = handlers
        self._running = False
        self._watchdog: Optional[WatchdogThread] = None
    
    def start(self) -> None:
        if not self._connection._connect():
            logger.error("[REDIS_BROKER] Cannot start consumer, not connected")
            return
        
        self._connection.ensure_consumer_group()
        self._running = True
        
        self._watchdog = WatchdogThread(
            target=self._consume_loop,
            name="redis-reading-consumer",
            max_restarts=10,
            backoff_seconds=5.0,
        )
        self._watchdog.start()
        
        logger.info(
            "[REDIS_BROKER] Consumer watchdog started: group=%s consumer=%s",
            RedisBrokerConnection.CONSUMER_GROUP,
            self._connection.consumer_name,
        )
    
    def _consume_loop(self) -> None:
        while self._running:
            try:
                if not self._connection.is_connected and not self._connection._connect():
                    time.sleep(5)
                    continue
                
                messages = self._connection.redis_client.xreadgroup(
                    groupname=RedisBrokerConnection.CONSUMER_GROUP,
                    consumername=self._connection.consumer_name,
                    streams={RedisBrokerConnection.STREAM_READINGS: ">"},
                    count=10,
                    block=1000,
                )
                
                if not messages:
                    continue
                
                for stream_name, stream_messages in messages:
                    for msg_id, fields in stream_messages:
                        try:
                            reading = self._parse_reading(fields)
                            for handler in self._handlers:
                                try:
                                    handler(reading)
                                except Exception as e:
                                    logger.exception("[REDIS_BROKER] Handler error: %s", str(e))
                            
                            self._connection.redis_client.xack(
                                RedisBrokerConnection.STREAM_READINGS,
                                RedisBrokerConnection.CONSUMER_GROUP,
                                msg_id,
                            )
                        except Exception as e:
                            logger.exception("[REDIS_BROKER] Message processing error: msg_id=%s", msg_id)
                
            except redis.ConnectionError as e:
                self._connection._connected = False
                logger.error("[REDIS_BROKER] Connection lost: %s. Reconnecting...", str(e))
                time.sleep(1)
            except Exception as e:
                logger.exception("[REDIS_BROKER] Consumer error: %s", str(e))
                time.sleep(1)
        
        logger.info("[REDIS_BROKER] Consumer stopped")
    
    def _parse_reading(self, fields: dict) -> Reading:
        def decode(v):
            return v.decode() if isinstance(v, bytes) else str(v)
        from ..reading_broker import Reading
        return Reading(
            sensor_id=int(decode(fields.get(b"sensor_id", fields.get("sensor_id", 0)))),
            sensor_type=decode(fields.get(b"sensor_type", fields.get("sensor_type", "unknown"))),
            value=float(decode(fields.get(b"value", fields.get("value", 0)))),
            timestamp=float(decode(fields.get(b"timestamp", fields.get("timestamp", 0)))),
        )
    
    def stop(self) -> None:
        self._running = False
        if self._watchdog:
            self._watchdog.stop()
    
    def is_healthy(self) -> bool:
        return self._watchdog is not None and self._watchdog.is_healthy()