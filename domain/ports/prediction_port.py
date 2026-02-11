"""Port de predicción — contrato que todo motor de predicción debe cumplir.

Este port vive en el dominio y NO conoce implementaciones concretas.
Los adaptadores (Taylor, Baseline, Ensemble) lo implementan en infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.prediction import Prediction
from ..entities.sensor_reading import SensorWindow


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
        """Genera predicción a partir de una ventana de lecturas.

        Args:
            window: Ventana temporal con lecturas del sensor.

        Returns:
            ``Prediction`` del dominio con valor, confianza, trend y metadata.

        Raises:
            ValueError: Si la ventana está vacía o contiene datos inválidos.
        """
        ...

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
