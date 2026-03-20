"""Kalman filter adapter for signal preprocessing.

Wraps KalmanSignalFilter for use with domain entities (SensorWindow).

This adapter is NOT deprecated — Kalman filtering is still a valid
preprocessing step for noisy sensor signals.
"""

from __future__ import annotations

import logging
from typing import Optional

from iot_machine_learning.domain.entities.sensor_reading import SensorReading, SensorWindow
from ..filters.kalman_filter import KalmanSignalFilter

logger = logging.getLogger(__name__)


class KalmanFilterAdapter:
    """Adapter que expone KalmanSignalFilter para uso en el dominio.

    No implementa PredictionPort porque Kalman es un filtro de señal,
    no un predictor.

    Attributes:
        _filters: Dict[sensor_id, KalmanSignalFilter] para estado por sensor.
        _Q: Ruido del proceso (process noise variance).
        _warmup_size: Número de puntos para warmup del filtro.
    """

    def __init__(
        self,
        Q: float = 1e-5,
        R: float = 0.01,
        warmup_size: int = 5,
    ) -> None:
        self._Q = Q
        self._R = R
        self._warmup_size = warmup_size
        self._filters: dict[int, KalmanSignalFilter] = {}

    def filter_window(self, window: SensorWindow) -> SensorWindow:
        """Filtra una ventana de señal usando Kalman.

        Args:
            window: Ventana con ruido.

        Returns:
            Nueva ventana con valores filtrados (timestamps sin modificar).
        """
        sensor_id = window.sensor_id
        values = window.values
        timestamps = window.timestamps

        if sensor_id not in self._filters:
            self._filters[sensor_id] = KalmanSignalFilter(
                Q=self._Q,
                R=self._R,
                warmup_size=self._warmup_size,
            )

        kalman = self._filters[sensor_id]
        filtered_values = [kalman.filter_next(v) for v in values]

        filtered_readings = [
            SensorReading(
                sensor_id=sensor_id,
                value=fv,
                timestamp=ts,
            )
            for fv, ts in zip(filtered_values, timestamps)
        ]

        return SensorWindow(sensor_id=sensor_id, readings=filtered_readings)

    def reset(self, sensor_id: Optional[int] = None) -> None:
        """Reinicia el filtro de un sensor (o todos).

        Args:
            sensor_id: ID del sensor a resetear. Si es None, resetea todos.
        """
        if sensor_id is None:
            self._filters.clear()
            logger.info("kalman_filters_all_reset")
        elif sensor_id in self._filters:
            del self._filters[sensor_id]
            logger.info("kalman_filter_reset", extra={"sensor_id": sensor_id})
