"""Adapter: Envuelve ml.baseline.predict_moving_average como PredictionPort.

Patrón Adapter (GoF): convierte la interfaz legacy de ml.baseline
a domain.ports.PredictionPort sin modificar el código original.

Esto permite:
1. Usar baseline en la arquitectura hexagonal enterprise.
2. No romper el código legacy de ml_service/.
3. Migración gradual: cuando baseline se reescriba, este adapter se elimina.
"""

from __future__ import annotations

import logging
from typing import Optional

from ....domain.entities.prediction import Prediction
from ....domain.entities.sensor_reading import SensorWindow
from ....domain.ports.prediction_port import PredictionPort

logger = logging.getLogger(__name__)


class BaselinePredictionAdapter(PredictionPort):
    """Adapter que expone predict_moving_average como PredictionPort.

    Attributes:
        _window: Tamaño de ventana para media móvil.
    """

    def __init__(self, window: int = 60) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "baseline_moving_average"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= 1

    def predict(self, window: SensorWindow) -> Prediction:
        """Adapta SensorWindow → List[float] y (value, conf) → Prediction.

        Args:
            window: Ventana temporal del sensor (entidad de dominio).

        Returns:
            ``Prediction`` (entidad de dominio).
        """
        from iot_machine_learning.infrastructure.ml.engines.baseline_engine import BaselineConfig, predict_moving_average

        values = window.values
        if not values:
            raise ValueError(f"Ventana vacía para sensor {window.sensor_id}")

        cfg = BaselineConfig(window=self._window)
        predicted_value, confidence = predict_moving_average(values, cfg)

        # Trend simple basado en últimos 2 valores
        trend = "stable"
        if len(values) >= 2:
            diff = values[-1] - values[-2]
            if diff > 0.01:
                trend = "up"
            elif diff < -0.01:
                trend = "down"

        return Prediction(
            sensor_id=window.sensor_id,
            predicted_value=predicted_value,
            confidence_score=confidence,
            trend=trend,
            engine_name="baseline_moving_average",
            metadata={
                "window": self._window,
                "n_points": len(values),
            },
        )

    def supports_confidence_interval(self) -> bool:
        return False
