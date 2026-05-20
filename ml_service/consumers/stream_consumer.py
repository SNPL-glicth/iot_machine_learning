"""ML Stream Consumer — reads readings:raw, triggers predictions.

At-least-once semantics: ACK after successful prediction processing.
MIGRATED 2026-04-09: Now uses RedisConnectionManager for centralized connection.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from iot_machine_learning.infrastructure.persistence.redis import RedisConnectionManager
from .sliding_window import SlidingWindowStore
from .prediction_worker import PredictionWorker, PredictionTask
from .backpressure import BackpressureController, DEFAULT_MAX_IN_FLIGHT
from .stream_predictor import predict_sensor, parse_reading, build_sensor_window
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

class ReadingsStreamConsumer:
    """Consumes readings:raw Redis Stream and triggers ML predictions."""

    def __init__(
        self,
        redis_url: Optional[str] = None,  # DEPRECATED: Now uses RedisConnectionManager
        min_window: int = DEFAULT_MIN_WINDOW,
        max_window: int = DEFAULT_MAX_WINDOW,
        consumer_name: Optional[str] = None,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        engine_factory: Optional[Callable[[], "PredictionAdapter"]] = None,
        db_engine: Optional["Engine"] = None,
        prediction_worker: Optional[PredictionWorker] = None,
        store: Optional[SlidingWindowStore] = None,
        tsdb_adapter=None,
        distributed_adapter=None,
    ):
        self._min_window = min_window
        self._consumer_name = consumer_name or f"ml_stream_{os.getpid()}"
        self._store = store or SlidingWindowStore(max_size=max_window)
        self._use_case = None
        self._running = False
        self._engine_factory = engine_factory
        self._db_engine = db_engine
        self._prediction_worker = prediction_worker
        self._tsdb = tsdb_adapter
        self._distributed = distributed_adapter
        self._migration_attempted: set = set()
        self._backpressure = BackpressureController(max_in_flight=max_in_flight)
    def _init_use_case(self):
        if self._use_case is not None:
            return
        try:
            from ..config.feature_flags import get_feature_flags
            from ..runners.wiring.container import BatchEnterpriseContainer
            engine = self._db_engine
            if engine is None:
                from iot_machine_learning.infrastructure.persistence.sql import ZeninDbConnection
                engine = ZeninDbConnection.get_engine()
            container = BatchEnterpriseContainer(engine=engine, flags=get_feature_flags())
            container.get_prediction_adapter()
            self._use_case = container
            logger.info("[STREAM_CONSUMER] Use case initialized")
        except Exception as e:
            logger.error("[STREAM_CONSUMER] Use case init failed: %s", e)

    def _connect_redis(self):
        self._redis = RedisConnectionManager.get_stream_client()
        self._redis.ping()
        try:
            self._redis.xgroup_create(STREAM_NAME, CONSUMER_GROUP, id="0", mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise
        logger.info("[STREAM_CONSUMER] Connected via RedisConnectionManager, group=%s", CONSUMER_GROUP)
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
                reading = parse_reading(fields)
                if reading is None:
                    self._redis.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                    continue
                window_size = self._store.append(reading)
                n_msgs += 1
                if self._tsdb is not None:
                    try:
                        self._tsdb.append(reading.sensor_id, reading.value, reading.timestamp)
                    except Exception as e:
                        logger.debug("[P3-5] tsdb_append_failed sensor=%d: %s", reading.sensor_id, e)

                if window_size >= self._min_window:
                    pending_acks.append((msg_id, reading.sensor_id))
                else:
                    self._redis.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)

        if self._prediction_worker is not None:
            for msg_id, sensor_id in pending_acks:
                enqueued = self._prediction_worker.enqueue(
                    PredictionTask(sensor_id=sensor_id)
                )
                if enqueued:
                    self._redis.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                else:
                    logger.warning(
                        "[STREAM_CONSUMER] prediction_worker_queue_full sensor=%d",
                        sensor_id,
                    )
        elif pending_acks:
            max_w = min(32, len(pending_acks))
            with ThreadPoolExecutor(max_workers=max_w) as executor:
                futures = {
                    executor.submit(self._predict, sensor_id): (msg_id, sensor_id)
                    for msg_id, sensor_id in pending_acks
                }
                for future in as_completed(futures, timeout=30):
                    msg_id, sensor_id = futures[future]
                    try:
                        future.result()
                        self._redis.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                    except Exception as e:
                        logger.warning(
                            {"event": "stream_batch_item_failed",
                             "sensor_id": sensor_id, "error": str(e)}
                        )

        batch_ms = (time.monotonic() - _t0) * 1000
        MetricsCollector.get_instance().record_reading_processed(batch_ms)
        logger.debug(
            "stream_batch ms=%.1f msgs=%d triggered=%d",
            batch_ms, n_msgs, len(pending_acks),
        )

    def _build_sensor_window(self, sensor_id: int):
        return build_sensor_window(
            sensor_id, self._store, self._min_window,
            self._distributed, self._migration_attempted,
        )

    def _predict(self, sensor_id: int) -> None:
        predict_sensor(
            self._use_case, sensor_id, self._store,
            self._min_window, self._distributed, self._migration_attempted,
        )
