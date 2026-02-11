"""Adapter: Envuelve TaylorPredictionEngine (Fase 1) como PredictionPort (Enterprise).

Patrón Adapter (GoF): convierte la interfaz de ml.core.PredictionEngine
a domain.ports.PredictionPort sin modificar el código original.

Esto permite:
1. Usar Taylor en la arquitectura hexagonal enterprise.
2. No romper el código legacy de ml/core/.
3. Migración gradual: cuando Taylor se reescriba para PredictionPort
   nativo, este adapter se elimina.

Mapeo de interfaces:
    PredictionEngine.predict(values, timestamps) → PredictionResult
    PredictionPort.predict(window: SensorWindow)  → Prediction
"""

from __future__ import annotations

import logging
from typing import Optional

from ....domain.entities.prediction import Prediction, PredictionConfidence
from ....domain.entities.sensor_reading import SensorWindow
from ....domain.ports.prediction_port import PredictionPort

logger = logging.getLogger(__name__)


class TaylorPredictionAdapter(PredictionPort):
    """Adapter que expone TaylorPredictionEngine como PredictionPort.

    Attributes:
        _engine: Instancia de TaylorPredictionEngine (Fase 1).
    """

    def __init__(
        self,
        order: int = 2,
        horizon: int = 1,
    ) -> None:
        """Inicializa el adapter creando un TaylorPredictionEngine.

        Args:
            order: Orden de Taylor (1–3).
            horizon: Pasos adelante a predecir.
        """
        from iot_machine_learning.infrastructure.ml.engines.taylor_engine import TaylorPredictionEngine
        self._engine = TaylorPredictionEngine(order=order, horizon=horizon)

    @property
    def name(self) -> str:
        return "taylor_adapted"

    def can_handle(self, n_points: int) -> bool:
        return self._engine.can_handle(n_points)

    def predict(self, window: SensorWindow) -> Prediction:
        """Adapta SensorWindow → List[float] y PredictionResult → Prediction.

        Args:
            window: Ventana temporal del sensor (entidad de dominio).

        Returns:
            ``Prediction`` (entidad de dominio).
        """
        values = window.values
        timestamps = window.timestamps if window.timestamps else None

        # Llamar al engine legacy
        result = self._engine.predict(values, timestamps)

        # Mapear PredictionResult → Prediction
        return Prediction(
            series_id=str(window.sensor_id),
            predicted_value=result.predicted_value,
            confidence_score=result.confidence,
            trend=result.trend,
            engine_name="taylor_adapted",
            metadata=result.metadata,
        )


class KalmanFilterAdapter:
    """Adapter que expone KalmanSignalFilter para uso en el dominio.

    No implementa PredictionPort porque Kalman es un filtro de señal,
    no un predictor.  Se usa como pre-procesamiento antes de predecir.

    Attributes:
        _filter: Instancia de KalmanSignalFilter (Fase 1).
    """

    def __init__(
        self,
        Q: float = 1e-5,
        warmup_size: int = 10,
    ) -> None:
        from iot_machine_learning.infrastructure.ml.filters.kalman_filter import KalmanSignalFilter
        self._filter = KalmanSignalFilter(Q=Q, warmup_size=warmup_size)

    def filter_window(self, window: SensorWindow) -> SensorWindow:
        """Filtra una ventana de sensor con Kalman.

        Args:
            window: Ventana original.

        Returns:
            Nueva ``SensorWindow`` con valores filtrados.
        """
        from ....domain.entities.sensor_reading import SensorReading, SensorWindow as SW

        filtered_values = [
            self._filter.filter_value(str(window.sensor_id), v)
            for v in window.values
        ]

        readings = [
            SensorReading(
                sensor_id=window.sensor_id,
                value=v,
                timestamp=t,
            )
            for v, t in zip(filtered_values, window.timestamps)
        ]

        return SW(sensor_id=window.sensor_id, readings=readings)

    def reset(self, sensor_id: Optional[int] = None) -> None:
        """Resetea estado del filtro."""
        self._filter.reset(str(sensor_id) if sensor_id is not None else None)
