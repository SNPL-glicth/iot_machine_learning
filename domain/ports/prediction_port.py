"""Port de predicción — contrato que todo motor de predicción debe cumplir.

Este port vive en el dominio y NO conoce implementaciones concretas.
Los adaptadores (Taylor, Baseline, Ensemble) lo implementan en infrastructure.

Dual interface: acepta ``SensorWindow`` (legacy IoT) o ``TimeSeries`` (agnóstico).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from ..entities.prediction import Prediction
from ..entities.sensor_reading import SensorWindow
from ..entities.time_series import TimeSeries


class PredictionPort(ABC):
    """Contrato para motores de predicción.

    Toda implementación debe:
    1. Retornar ``Prediction`` del dominio (no PredictionResult de ml.core).
    2. Manejar edge cases (pocos datos, NaN) sin crashear.
    3. Proveer ``name`` para logging/auditoría.
    4. Indicar si puede operar con N puntos vía ``can_handle``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificador del motor."""
        ...

    @abstractmethod
    def predict(self, window: SensorWindow) -> Prediction:
        """Genera predicción a partir de una ventana de lecturas (legacy).

        Args:
            window: Ventana temporal con lecturas del sensor.

        Returns:
            ``Prediction`` del dominio con valor, confianza, trend y metadata.

        Raises:
            ValueError: Si la ventana está vacía o contiene datos inválidos.
        """
        ...

    def predict_series(self, series: TimeSeries) -> Prediction:
        """Genera predicción a partir de una TimeSeries agnóstica.

        Implementación por defecto delega a ``predict`` creando un
        ``SensorWindow`` temporal.  Motores nuevos pueden override
        para operar directamente sobre ``TimeSeries``.

        Args:
            series: Serie temporal agnóstica.

        Returns:
            ``Prediction`` del dominio.
        """
        from ..entities.sensor_reading import SensorReading

        readings = [
            SensorReading(
                sensor_id=int(series.series_id) if series.series_id.isdigit() else 0,
                value=p.v,
                timestamp=p.t,
            )
            for p in series.points
        ]
        sw = SensorWindow(
            sensor_id=int(series.series_id) if series.series_id.isdigit() else 0,
            readings=readings,
        )
        return self.predict(sw)

    @abstractmethod
    def can_handle(self, n_points: int) -> bool:
        """Indica si el motor puede operar con ``n_points`` datos.

        Args:
            n_points: Número de puntos disponibles.

        Returns:
            ``True`` si puede generar predicción válida.
        """
        ...

    def supports_confidence_interval(self) -> bool:
        """Indica si el motor provee intervalos de confianza."""
        return False
