"""Port de detección de anomalías — contrato para detectores.

Desacopla el dominio de implementaciones concretas (IsolationForest,
LOF, Z-score, Voting Ensemble, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..entities.anomaly import AnomalyResult
from ..entities.sensor_reading import SensorWindow


class AnomalyDetectionPort(ABC):
    """Contrato para detectores de anomalías.

    Toda implementación debe:
    1. Retornar ``AnomalyResult`` del dominio.
    2. Soportar entrenamiento con datos históricos.
    3. Proveer score normalizado (0–1).
    4. Incluir explicación legible en el resultado.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del detector (para logging/auditoría)."""
        ...

    @abstractmethod
    def train(self, historical_values: List[float]) -> None:
        """Entrena el detector con datos históricos.

        Args:
            historical_values: Serie temporal de entrenamiento.

        Raises:
            ValueError: Si no hay suficientes datos.
        """
        ...

    @abstractmethod
    def detect(self, window: SensorWindow) -> AnomalyResult:
        """Detecta anomalías en la ventana actual.

        Args:
            window: Ventana temporal del sensor.

        Returns:
            ``AnomalyResult`` con score, votos, explicación.
        """
        ...

    @abstractmethod
    def is_trained(self) -> bool:
        """``True`` si el detector fue entrenado."""
        ...
