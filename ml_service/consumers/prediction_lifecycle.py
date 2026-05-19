"""P3 lifecycle: wire PredictionWorker, ResultStore, and StreamConsumer.

Environment variables:
  ML_PREDICTION_WORKERS — worker threads (default: 4)
  ML_PREDICTION_QUEUE_MAX — queue capacity (default: 5000)
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _build_sensor_window(store, sensor_id: int, min_window: int = 5):
    from iot_machine_learning.domain.entities.sensor_reading import (
        SensorReading, SensorWindow,
    )
    readings_raw = store.get_window(sensor_id)
    if len(readings_raw) < min_window:
        return None
    readings = [
        SensorReading(sensor_id=sensor_id, value=r.value, timestamp=r.timestamp)
        for r in readings_raw
        if r.value == r.value and r.timestamp > 0
    ]
    if len(readings) < 2:
        return None
    return SensorWindow(sensor_id=sensor_id, readings=readings)


def _make_worker_factory(store):
    """Return a factory that creates a per-worker predict function."""
    from ..config.feature_flags import get_feature_flags
    from ..runners.wiring.container import BatchEnterpriseContainer
    from ..api.services.prediction_service import PredictionService

    def factory():
        from ...infrastructure.persistence.sql.zenin_db_connection import ZeninDbConnection
        engine = ZeninDbConnection.get_engine()
        flags = get_feature_flags()
        container = BatchEnterpriseContainer(engine=engine, flags=flags)
        stream_adapter = container.get_prediction_adapter()

        def predict_fn(task):
            if task.mode == "stream":
                window = _build_sensor_window(store, task.sensor_id)
                if window is None:
                    return None
                return stream_adapter.predict_with_window(sensor_window=window)
            # HTTP mode — inject orchestrator when flag enabled (same pattern as routes.py)
            cognitive_orchestrator = None
            if flags.ML_USE_COGNITIVE_ORCHESTRATOR:
                try:
                    cognitive_adapter = container.get_cognitive_adapter()
                    cognitive_orchestrator = cognitive_adapter.orchestrator
                    logger.info(
                        "cognitive_orchestrator_injected",
                        extra={"caller": "prediction_lifecycle", "sensor_id": task.sensor_id},
                    )
                except Exception as exc:
                    logger.warning(
                        "cognitive_orchestrator_init_failed",
                        extra={
                            "caller": "prediction_lifecycle",
                            "error": str(exc),
                            "fallback": "baseline+kalman",
                            "sensor_id": task.sensor_id,
                        },
                    )
            with engine.connect() as conn:
                service = PredictionService(
                    conn, cognitive_orchestrator=cognitive_orchestrator,
                )
                return service.predict(
                    sensor_id=task.sensor_id,
                    horizon_minutes=task.horizon_minutes,
                    window=task.window,
                    dedupe_minutes=task.dedupe_minutes,
                )
        return predict_fn
    return factory


def init_prediction_worker(app: Any) -> None:
    from .prediction_worker import PredictionWorker
    from ..api.result_store import PredictionResultStore
    from .sliding_window import SlidingWindowStore
    from .stream_consumer import ReadingsStreamConsumer

    result_store = PredictionResultStore()
    stream_store = SlidingWindowStore(max_size=20)

    worker = PredictionWorker(
        n_workers=int(os.environ.get("ML_PREDICTION_WORKERS", "4")),
        queue_max=int(os.environ.get("ML_PREDICTION_QUEUE_MAX", "5000")),
        worker_factory=_make_worker_factory(stream_store),
        result_store=result_store,
    )
    worker.start()

    stream_consumer = ReadingsStreamConsumer(
        store=stream_store,
        prediction_worker=worker,
    )
    t = threading.Thread(target=stream_consumer.start, daemon=True)
    t.start()

    app.state.prediction_worker = worker
    app.state.result_store = result_store
    logger.info("[P3] PredictionWorker + StreamConsumer started")


def stop_prediction_worker(app: Any) -> None:
    worker = getattr(app.state, "prediction_worker", None)
    if worker is not None:
        try:
            worker.stop(timeout=30.0)
            logger.info("[P3] PredictionWorker stopped")
        except Exception as e:
            logger.warning("[P3] PredictionWorker stop failed: %s", e)
