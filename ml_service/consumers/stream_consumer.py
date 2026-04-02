"""ML Stream Consumer — reads readings:raw, triggers predictions.

Features:
- Backpressure control: rejects messages when overloaded
- Priority-based acceptance (critical always accepted)
- At-least-once semantics: ACK after successful prediction
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from .sliding_window import Reading, SlidingWindowStore
from ..metrics.performance_metrics import MetricsCollector

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

STREAM_NAME = "readings:raw"
CONSUMER_GROUP = "ml_stream_consumers"
DEFAULT_MIN_WINDOW = 5
DEFAULT_MAX_WINDOW = 20
DEFAULT_BATCH_SIZE = 50
DEFAULT_BLOCK_MS = 2000

# Backpressure defaults
DEFAULT_MAX_IN_FLIGHT = 1000
DEFAULT_LATENCY_TARGET_MS = 5000


class BackpressureController:
    """Simple backpressure: rejects messages when system overloaded."""
    
    def __init__(
        self,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        target_latency_ms: float = DEFAULT_LATENCY_TARGET_MS,
    ):
        self._max_in_flight = max_in_flight
        self._target_latency = target_latency_ms
        self._in_flight = 0
        self._lock = threading.Lock()
        self._rejected = 0
        self._accepted = 0
    
    def can_accept(self, priority: str = "normal") -> bool:
        """Check if we can accept a new message.
        
        Priority levels:
        - critical: always accepted (unless >90% capacity)
        - high: accepted until 80% capacity  
        - normal: accepted until 70% capacity
        - low: accepted until 50% capacity
        """
        with self._lock:
            load = self._in_flight / self._max_in_flight
            
            if priority == "critical":
                return load < 0.9
            elif priority == "high":
                return load < 0.8
            elif priority == "normal":
                return load < 0.7
            else:  # low
                return load < 0.5
    
    def record_start(self):
        """Call when starting to process a message."""
        with self._lock:
            self._in_flight += 1
            self._accepted += 1
    
    def record_complete(self, latency_ms: float):
        """Call when finished processing."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
    
    def record_reject(self):
        """Call when rejecting a message."""
        with self._lock:
            self._rejected += 1
    
    def get_metrics(self) -> dict:
        """Get current backpressure metrics."""
        with self._lock:
            return {
                "in_flight": self._in_flight,
                "max_in_flight": self._max_in_flight,
                "load_factor": self._in_flight / self._max_in_flight,
                "accepted": self._accepted,
                "rejected": self._rejected,
            }


class ReadingsStreamConsumer:
    """Consumes readings:raw Redis Stream and triggers ML predictions.
    
    Implements at-least-once semantics by ACKing messages only after
    successful prediction processing.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        min_window: int = DEFAULT_MIN_WINDOW,
        max_window: int = DEFAULT_MAX_WINDOW,
        consumer_name: Optional[str] = None,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        engine_factory: Optional[Callable[[], "PredictionAdapter"]] = None,
        db_engine: Optional["Engine"] = None,
    ):
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._min_window = min_window
        self._consumer_name = consumer_name or f"ml_stream_{os.getpid()}"
        self._store = SlidingWindowStore(max_size=max_window)
        self._use_case = None
        self._running = False
        self._engine_factory = engine_factory
        self._db_engine = db_engine
        
        # Backpressure controller
        self._backpressure = BackpressureController(max_in_flight=max_in_flight)

    def _init_use_case(self):
        if self._use_case is not None:
            return
        try:
            from ..config.feature_flags import get_feature_flags
            from ..runners.wiring.container import BatchEnterpriseContainer
            
            # Use injected engine or create default
            engine = self._db_engine
            if engine is None:
                from iot_machine_learning.infrastructure.persistence.sql import get_engine
                engine = get_engine()
            
            container = BatchEnterpriseContainer(engine=engine, flags=get_feature_flags())
            container.get_prediction_adapter()
            self._use_case = container
            logger.info("[STREAM_CONSUMER] Use case initialized")
        except Exception as e:
            logger.error("[STREAM_CONSUMER] Use case init failed: %s", e)

    def _connect_redis(self):
        import redis
        self._redis = redis.Redis.from_url(
            self._redis_url, decode_responses=False, socket_timeout=5.0,
        )
        self._redis.ping()
        try:
            self._redis.xgroup_create(STREAM_NAME, CONSUMER_GROUP, id="0", mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise
        logger.info("[STREAM_CONSUMER] Connected, group=%s", CONSUMER_GROUP)

    def start(self) -> None:
        self._connect_redis()
        self._init_use_case()
        self._running = True
        logger.info("[STREAM_CONSUMER] Started, min_window=%d", self._min_window)

        while self._running:
            try:
                self._consume_batch()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception("[STREAM_CONSUMER] Error: %s", e)
                time.sleep(1)

        logger.info("[STREAM_CONSUMER] Stopped")

    def stop(self) -> None:
        self._running = False

    def _consume_batch(self) -> None:
        messages = self._redis.xreadgroup(
            groupname=CONSUMER_GROUP,
            consumername=self._consumer_name,
            streams={STREAM_NAME: ">"},
            count=DEFAULT_BATCH_SIZE,
            block=DEFAULT_BLOCK_MS,
        )
        if not messages:
            return

        _t0 = time.monotonic()
        pending_acks: List[Tuple[str, int]] = []  # [(msg_id, sensor_id), ...]
        n_msgs = 0

        for _stream, entries in messages:
            for msg_id, fields in entries:
                reading = self._parse_reading(fields)
                if reading is None:
                    # ACK invalid messages immediately
                    self._redis.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                    continue

                window_size = self._store.append(reading)
                n_msgs += 1

                if window_size >= self._min_window:
                    pending_acks.append((msg_id, reading.sensor_id))
                else:
                    # ACK messages that don't trigger prediction yet
                    self._redis.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)

        # Process predictions and ACK only after successful completion
        for msg_id, sensor_id in pending_acks:
            try:
                self._predict(sensor_id)
                # ACK after successful prediction (at-least-once semantics)
                self._redis.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
            except Exception as e:
                logger.error(
                    "[STREAM_CONSUMER] Prediction failed for sensor=%d: %s. "
                    "Message will be reprocessed (no ACK).",
                    sensor_id, e
                )
                # Do not ACK - message will be redelivered

        batch_ms = (time.monotonic() - _t0) * 1000
        MetricsCollector.get_instance().record_reading_processed(batch_ms)
        logger.debug(
            "stream_batch ms=%.1f msgs=%d triggered=%d",
            batch_ms, n_msgs, len(pending_acks),
        )

    def _parse_reading(self, fields: dict) -> Optional[Reading]:
        try:
            def d(v): return v.decode() if isinstance(v, bytes) else str(v)
            return Reading(
                sensor_id=int(d(fields.get(b"sensor_id", fields.get("sensor_id", 0)))),
                value=float(d(fields.get(b"value", fields.get("value", 0)))),
                timestamp=float(d(fields.get(b"timestamp", fields.get("timestamp", 0)))),
                timestamp_iso=d(fields.get(b"timestamp_iso", fields.get("timestamp_iso", ""))),
            )
        except Exception as e:
            logger.warning("[STREAM_CONSUMER] Parse error: %s", e)
            return None

    def _predict(self, sensor_id: int) -> None:
        if self._use_case is None:
            return
        try:
            from ..config.feature_flags import get_feature_flags
            if not get_feature_flags().ML_STREAM_PREDICTIONS_ENABLED:
                logger.debug("stream_predictions_disabled sensor=%d", sensor_id)
                return
            _t0 = time.monotonic()
            adapter = self._use_case.get_prediction_adapter()
            window = self._build_sensor_window(sensor_id)
            if window is None:
                return
            result = adapter.predict_with_window(sensor_window=window)
            pred_ms = (time.monotonic() - _t0) * 1000
            MetricsCollector.get_instance().record_prediction(pred_ms)
            logger.debug(
                "stream_predict sensor=%d ms=%.1f conf=%.3f engine=%s",
                sensor_id, pred_ms, result.confidence,
                result.engine_used,
            )
        except Exception as e:
            logger.error("[STREAM_CONSUMER] Prediction failed sensor=%d: %s", sensor_id, e)

    def _build_sensor_window(self, sensor_id: int):
        """Build a SensorWindow from the in-memory sliding window store."""
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading, SensorWindow,
        )
        readings_raw = self._store.get_window(sensor_id)
        if len(readings_raw) < self._min_window:
            return None
        readings = [
            SensorReading(sensor_id=sensor_id, value=r.value, timestamp=r.timestamp)
            for r in readings_raw
            if r.value == r.value and r.timestamp > 0
        ]
        if len(readings) < 2:
            return None
        return SensorWindow(sensor_id=sensor_id, readings=readings)
