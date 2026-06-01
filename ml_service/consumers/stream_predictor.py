"""Stream prediction dispatcher — predict + build sensor window.

Extracted from stream_consumer.py for modularization.
Handles prediction triggering and window migration from distributed replicas.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from .sliding_window import Reading, SlidingWindowStore
from ..metrics.performance_metrics import MetricsCollector
from ...domain.validators.data_sanitizer import DataSanitizer

logger = logging.getLogger(__name__)
_MIN_ML_POINTS = DataSanitizer.MIN_POINTS  # 3


def parse_reading(fields: dict) -> Optional[Reading]:
    """Decode Redis stream fields into a Reading object.
    FIX B6: emit WARNING for None/missing coercions (never silent)."""
    try:
        def d(v): return v.decode() if isinstance(v, bytes) else str(v)
        raw_sid = fields.get(b"sensor_id", fields.get("sensor_id"))
        raw_val = fields.get(b"value", fields.get("value"))
        raw_ts = fields.get(b"timestamp", fields.get("timestamp"))
        if raw_sid is None or raw_val is None or raw_ts is None:
            logger.warning(
                {"event": "ingest_none_coercion",
                 "sensor_id": str(raw_sid), "value": str(raw_val),
                 "timestamp": str(raw_ts), "coerced_to": 0.0}
            )
        sensor_id = int(d(raw_sid)) if raw_sid is not None else 0
        value = float(d(raw_val)) if raw_val is not None else 0.0
        timestamp = float(d(raw_ts)) if raw_ts is not None else 0.0
        return Reading(
            sensor_id=sensor_id, value=value, timestamp=timestamp,
        )
    except Exception as e:
        logger.warning("[STREAM_CONSUMER] Parse error: %s", e)
        return None


def predict_sensor(use_case, sensor_id: int, store: SlidingWindowStore,
                   min_window: int, distributed_adapter,
                   migration_attempted: set) -> None:
    """Execute prediction for a sensor using its current window.

    Args:
        use_case: BatchEnterpriseContainer with get_prediction_adapter()
        sensor_id: Target sensor
        store: SlidingWindowStore with buffered readings
        min_window: Minimum window size required
        distributed_adapter: Optional distributed window adapter
        migration_attempted: Set tracking sensors already migrated
    """
    if use_case is None:
        return
    try:
        from ..config.feature_flags import get_feature_flags
        # ARCHITECTURE NOTE (INF-5): stream predictions y batch runner
        # son paths MUTUAMENTE EXCLUYENTES. Habilitar ambos produce
        # filas duplicadas en dbo.predictions y eventos duplicados en
        # dbo.ml_events. El guard en lifespan previene esta configuración.
        if not get_feature_flags().ML_STREAM_PREDICTIONS_ENABLED:
            logger.debug("stream_predictions_disabled sensor=%d", sensor_id)
            return
        _t0 = time.monotonic()
        adapter = use_case.get_prediction_adapter()
        window = build_sensor_window(
            sensor_id, store, min_window, distributed_adapter, migration_attempted
        )
        if window is None:
            return
        result = adapter.predict_with_window(sensor_window=window)
        pred_ms = (time.monotonic() - _t0) * 1000
        MetricsCollector.get_instance().record_prediction(pred_ms)
        logger.debug(
            "stream_predict sensor=%d ms=%.1f conf=%.3f engine=%s",
            sensor_id, pred_ms, result.confidence, result.engine_used,
        )
    except Exception as e:
        logger.error("[STREAM_CONSUMER] Prediction failed sensor=%d: %s", sensor_id, e)


def build_sensor_window(sensor_id: int, store: SlidingWindowStore,
                        min_window: int, distributed_adapter,
                        migration_attempted: set) -> Optional[object]:
    """Build a SensorWindow from the in-memory sliding window store.

    FIX P3-6: Si la ventana local tiene < min_window puntos y
    distributed_adapter está disponible, intenta migrar desde Redis.
    """
    from iot_machine_learning.domain.entities.sensor_reading import (
        SensorReading, SensorWindow,
    )
    readings_raw = store.get_window(sensor_id)
    effective_min = max(min_window, _MIN_ML_POINTS)

    # FIX P3-6: intentar migración desde otra réplica UNA vez por sensor
    if (len(readings_raw) < effective_min
            and distributed_adapter is not None
            and sensor_id not in migration_attempted):
        migration_attempted.add(sensor_id)
        try:
            remote = distributed_adapter.get(sensor_id)
            if remote:
                logger.info("[P3-6] window_migrated sensor=%d remote=%d",
                            sensor_id, len(remote))
                for r in remote:
                    store.append(
                        Reading(sensor_id=sensor_id, value=r.value,
                                timestamp=r.timestamp)
                    )
                readings_raw = store.get_window(sensor_id)
        except Exception as e:
            logger.warning("[P3-6] window_migration_failed sensor=%d: %s", sensor_id, e)

    if len(readings_raw) < effective_min:
        logger.warning(
            "insufficient_data sensor=%d needed=%d got=%d",
            sensor_id, effective_min, len(readings_raw),
        )
        return None
    readings = [
        SensorReading(sensor_id=sensor_id, value=r.value, timestamp=r.timestamp)
        for r in readings_raw
        if r.value == r.value and r.timestamp > 0
    ]
    if len(readings) < 2:
        return None
    return SensorWindow(sensor_id=sensor_id, readings=readings)


# BACKLOG — Rediseños arquitecturales para 1000+ sensores:
#
# C1. SEPARAR INGESTIÓN DE PREDICCIÓN:
#   Hoy: mismo hilo lee Redis y predice (acoplamiento fuerte).
#   Target: asyncio.Queue interna. Thread A: consume Redis → queue.
#            Thread pool B: workers leen queue → predicción.
#   Beneficio: ingestión nunca bloqueada por predicción lenta.
#   Prerequisito: MLEventPersister thread-safe (B2 debe estar completo).
#
# C2. SHARD POR SENSOR (SlidingWindowStore):
#   Hoy: un store global → contención con 1000 sensores.
#   Target: consistent hashing sensor_id → shard index →
#            N stores independientes (N = workers count).
#   Beneficio: elimina lock contention en store.
#   Prerequisito: A2 (ThreadPoolExecutor) debe estar completo.
#
# C3. COGNITIVE ORCHESTRATOR OPT-IN POR SENSOR:
#   Hoy: ML_USE_COGNITIVE_ORCHESTRATOR es global (on/off para todos).
#   Target: tabla sensor_config con cognitive_enabled: bool por sensor.
#   Beneficio: sensores críticos usan orchestrator completo,
#              sensores simples usan pipeline ligero.
#   Prerequisito: schema migration en iot DB.
