"""Execution mixin for PredictSensorValueUseCase.

Provides execute_with_window for pre-loaded SensorWindow scenarios.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ...domain.entities.sensor_reading import SensorWindow
from ..dto.prediction_dto import PredictionDTO

logger = logging.getLogger(__name__)


class _PredictionExecutionMixin:
    """Mixin that provides execute_with_window public method."""

    def execute_with_window(self: Any, sensor_window: SensorWindow) -> PredictionDTO:
        """Execute prediction with a pre-loaded SensorWindow (no SQL reload)."""
        t_start = time.monotonic()
        if sensor_window.is_empty:
            raise ValueError(
                f"Sensor {sensor_window.sensor_id} no tiene lecturas disponibles"
            )

        try:
            sanitized = self._sanitizer.sanitize(
                values=sensor_window.values,
                timestamps=sensor_window.timestamps,
            )
        except ValueError as e:
            raise ValueError(
                f"Data validation failed for sensor {sensor_window.sensor_id}: {e}"
            ) from e

        if sanitized.warnings:
            logger.warning(
                "input_sanitization_warnings",
                extra={
                    "sensor_id": sensor_window.sensor_id,
                    "warnings": sanitized.warnings,
                },
            )

        sanitized_window = SensorWindow(
            sensor_id=sensor_window.sensor_id,
            values=sanitized.values,
            timestamps=sanitized.timestamps,
        )

        logger.info(
            "use_case_predict_preloaded_start",
            extra={
                "sensor_id": sensor_window.sensor_id,
                "window_size": sanitized_window.size,
                "sanitization_warnings": len(sanitized.warnings),
            },
        )
        prediction = self._prediction_service.predict(sanitized_window)
        self._persist(prediction, sensor_window.sensor_id)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        dto = self._build_dto(prediction)
        logger.info(
            "use_case_predict_preloaded_complete",
            extra={
                "sensor_id": sensor_window.sensor_id,
                "predicted_value": prediction.predicted_value,
                "engine": prediction.engine_name,
                "elapsed_ms": round(elapsed_ms, 2),
                "trace_id": prediction.audit_trace_id,
            },
        )
        return dto
