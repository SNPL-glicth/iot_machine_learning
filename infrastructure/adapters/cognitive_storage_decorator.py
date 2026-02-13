"""Decorator that adds cognitive memory dual-write to any StoragePort.

Wraps an existing ``StoragePort`` implementation (e.g. ``SqlServerStorageAdapter``)
and, after each successful transactional write, fires a corresponding
``remember_*`` call on the ``CognitiveMemoryPort``.

Design principles:
    - **SQL is source of truth.**  The inner storage always executes first.
      If it fails, the exception propagates normally — no cognitive write.
    - **Cognitive writes are fire-and-forget.**  If ``remember_*`` fails,
      the error is logged and swallowed.  The ML pipeline is never broken.
    - **Feature-flag gated.**  If ``ML_ENABLE_COGNITIVE_MEMORY`` is ``False``,
      the decorator is a transparent pass-through (zero overhead).
    - **Async-safe.**  When ``ML_COGNITIVE_MEMORY_ASYNC`` is ``True``,
      cognitive writes run in a background thread to avoid blocking the
      transactional path.
    - **No domain modification.**  This file lives in infrastructure and
      only depends on domain ports and entities.
    - **No SQL adapter modification.**  The inner ``StoragePort`` is
      untouched — pure composition via the Decorator pattern.

Usage (DI wiring)::

    from iot_machine_learning.infrastructure.adapters.cognitive_storage_decorator import (
        CognitiveStorageDecorator,
    )

    sql_storage = SqlServerStorageAdapter(conn)
    cognitive = WeaviateCognitiveAdapter(url, enabled=True)
    storage = CognitiveStorageDecorator(sql_storage, cognitive, flags)

    # Pass ``storage`` wherever ``StoragePort`` is expected.
    use_case = PredictSensorValueUseCase(prediction_service, storage)
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from ...domain.entities.anomaly import AnomalyResult
from ...domain.entities.prediction import Prediction
from ...domain.entities.sensor_reading import SensorWindow
from ...domain.entities.time_series import TimeSeries
from ...domain.ports.cognitive_memory_port import CognitiveMemoryPort
from ...domain.ports.storage_port import StoragePort
from ...ml_service.config.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


class CognitiveStorageDecorator(StoragePort):
    """Decorator that adds cognitive memory writes to a StoragePort.

    Delegates ALL operations to the inner ``StoragePort``.
    After ``save_prediction`` and ``save_anomaly_event`` succeed,
    fires the corresponding ``remember_*`` on the cognitive port.

    Args:
        inner: The real ``StoragePort`` (e.g. ``SqlServerStorageAdapter``).
        cognitive: A ``CognitiveMemoryPort`` implementation.
        flags: Feature flags controlling cognitive memory behaviour.
    """

    def __init__(
        self,
        inner: StoragePort,
        cognitive: CognitiveMemoryPort,
        flags: FeatureFlags,
    ) -> None:
        self._inner = inner
        self._cognitive = cognitive
        self._flags = flags

    # ------------------------------------------------------------------
    # Helper: fire cognitive write (sync or async, fail-safe)
    # ------------------------------------------------------------------

    def _fire_cognitive(self, fn_name: str, *args, **kwargs) -> None:
        """Call a cognitive method, optionally in a background thread.

        Never raises.  Logs and swallows all errors.
        """
        if not self._flags.ML_ENABLE_COGNITIVE_MEMORY:
            return

        def _do_call() -> None:
            try:
                method = getattr(self._cognitive, fn_name)
                result = method(*args, **kwargs)
                logger.debug(
                    "cognitive_dual_write_ok",
                    extra={"method": fn_name, "result": result},
                )
            except Exception as exc:
                logger.warning(
                    "cognitive_dual_write_error",
                    extra={"method": fn_name, "error": str(exc)},
                )

        if self._flags.ML_COGNITIVE_MEMORY_ASYNC:
            thread = threading.Thread(
                target=_do_call,
                name=f"cognitive-{fn_name}",
                daemon=True,
            )
            thread.start()
        else:
            _do_call()

    # ------------------------------------------------------------------
    # StoragePort: pass-through reads
    # ------------------------------------------------------------------

    def load_sensor_window(
        self,
        sensor_id: int,
        limit: int = 500,
    ) -> SensorWindow:
        return self._inner.load_sensor_window(sensor_id, limit)

    def list_active_sensor_ids(self) -> List[int]:
        return self._inner.list_active_sensor_ids()

    def get_latest_prediction(self, sensor_id: int) -> Optional[Prediction]:
        return self._inner.get_latest_prediction(sensor_id)

    def get_sensor_metadata(self, sensor_id: int) -> Dict[str, object]:
        return self._inner.get_sensor_metadata(sensor_id)

    def get_device_id_for_sensor(self, sensor_id: int) -> int:
        return self._inner.get_device_id_for_sensor(sensor_id)

    # ------------------------------------------------------------------
    # StoragePort: agnostic series_id pass-through reads
    # ------------------------------------------------------------------

    def load_series_window(
        self, series_id: str, limit: int = 500
    ) -> TimeSeries:
        return self._inner.load_series_window(series_id, limit)

    def list_active_series_ids(self) -> List[str]:
        return self._inner.list_active_series_ids()

    def get_latest_prediction_for_series(
        self, series_id: str
    ) -> Optional[Prediction]:
        return self._inner.get_latest_prediction_for_series(series_id)

    def get_series_metadata(self, series_id: str) -> Dict[str, object]:
        return self._inner.get_series_metadata(series_id)

    # ------------------------------------------------------------------
    # StoragePort: dual-write saves
    # ------------------------------------------------------------------

    def save_prediction(self, prediction: Prediction) -> int:
        """Save prediction to SQL, then fire cognitive remember_explanation."""
        # 1. SQL first (source of truth)
        record_id = self._inner.save_prediction(prediction)

        # 2. Cognitive dual-write (fire-and-forget)
        self._fire_cognitive(
            "remember_explanation",
            prediction,
            record_id,
            explanation_text=str(
                prediction.metadata.get("explanation", "")
            ),
            domain_name="iot",
        )

        return record_id

    def save_anomaly_event(
        self,
        anomaly: AnomalyResult,
        prediction_id: Optional[int] = None,
    ) -> int:
        """Save anomaly to SQL, then fire cognitive remember_anomaly."""
        # 1. SQL first (source of truth)
        record_id = self._inner.save_anomaly_event(anomaly, prediction_id)

        # 2. Cognitive dual-write (fire-and-forget)
        context_str = ""
        if anomaly.context:
            import json
            try:
                context_str = json.dumps(anomaly.context, default=str)
            except Exception:
                context_str = str(anomaly.context)

        self._fire_cognitive(
            "remember_anomaly",
            anomaly,
            record_id,
            event_code="ANOMALY_DETECTED",
            behavior_pattern="",
            operational_context=context_str,
            domain_name="iot",
        )

        return record_id
