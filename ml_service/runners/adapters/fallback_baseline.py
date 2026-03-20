"""Fallback baseline para batch runner enterprise bridge.

Cuando el enterprise stack falla, este módulo provee una predicción
baseline MEJORADA respecto al legacy:
- Carga SensorWindow CON timestamps (no solo valores)
- Filtra NULLs via _is_valid_sensor_value (no convierte NULL → 0.0)
- Orden cronológico ASC (no DESC)

Fix CRIT-3: NO convierte NULL → 0.0, usa StoragePort que descarta NULLs.

Restricción: < 180 líneas.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iot_machine_learning.domain.ports.storage_port import StoragePort

logger = logging.getLogger(__name__)


def fallback_to_baseline(
    storage: "StoragePort",
    sensor_id: int,
    window_size: int,
) -> "BatchPredictionResult":
    """Fallback a baseline MEJORADO (no el del batch runner legacy).

    Mejoras sobre el path legacy:
    - Carga via StoragePort (filtra NULLs, orden cronológico)
    - Usa predict_moving_average del engine baseline

    Args:
        storage: StoragePort para cargar SensorWindow.
        sensor_id: ID del sensor.
        window_size: Puntos a cargar.

    Returns:
        ``BatchPredictionResult`` con engine_used="baseline_fallback".
        NUNCA lanza excepción.
    """
    from .enterprise_prediction import BatchPredictionResult

    try:
        window = storage.load_sensor_window(
            sensor_id=sensor_id,
            limit=window_size,
        )

        if window.is_empty:
            logger.warning(
                "fallback_baseline_empty",
                extra={"sensor_id": sensor_id},
            )
            return BatchPredictionResult(
                predicted_value=0.0,
                confidence=0.0,
                trend="stable",
                engine_used="baseline_fallback_empty",
            )

        values = window.values

        from iot_machine_learning.infrastructure.ml.engines.baseline import (
            BaselineConfig,
            predict_moving_average,
        )

        cfg = BaselineConfig(window=len(values))
        predicted_value, confidence = predict_moving_average(values, cfg)

        # Trend simple basado en últimos 2 valores
        trend = "stable"
        if len(values) >= 2:
            diff = values[-1] - values[-2]
            if diff > 0.01:
                trend = "up"
            elif diff < -0.01:
                trend = "down"

        return BatchPredictionResult(
            predicted_value=predicted_value,
            confidence=confidence,
            trend=trend,
            engine_used="baseline_fallback",
        )

    except Exception as exc:
        logger.error(
            "fallback_baseline_error",
            extra={"sensor_id": sensor_id, "error": str(exc)},
        )
        return BatchPredictionResult(
            predicted_value=0.0,
            confidence=0.0,
            trend="stable",
            engine_used="baseline_fallback_error",
        )
