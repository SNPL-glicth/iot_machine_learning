"""Port de detección de patrones — contrato para detectores de patrones,
change points y clasificación de spikes.

Agrupa tres capacidades relacionadas bajo un solo port porque
comparten el mismo contexto de entrada (ventana de sensor).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.pattern import (
    ChangePoint,
    DeltaSpikeResult,
    OperationalRegime,
    PatternResult,
)
from ..entities.sensor_reading import SensorWindow


class PatternDetectionPort(ABC):
    """Contrato para detección de patrones de comportamiento."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del detector."""
        ...

    @abstractmethod
    def detect_pattern(self, window: SensorWindow) -> PatternResult:
        """Detecta el patrón de comportamiento actual.

        Args:
            window: Ventana temporal del sensor.

        Returns:
            ``PatternResult`` con tipo, confianza y descripción.
        """
        ...


class ChangePointDetectionPort(ABC):
    """Contrato para detección de puntos de cambio estructural."""

    @abstractmethod
    def detect_online(self, value: float) -> Optional[ChangePoint]:
        """Detecta cambio en modo online (1 valor a la vez).

        Args:
            value: Nuevo valor del sensor.

        Returns:
            ``ChangePoint`` si se detectó cambio, ``None`` si no.
        """
        ...

    @abstractmethod
    def detect_batch(self, values: List[float]) -> List[ChangePoint]:
        """Detecta cambios en batch (ventana completa).

        Args:
            values: Serie temporal completa.

        Returns:
            Lista de ``ChangePoint`` detectados.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reinicia estado interno del detector."""
        ...


class DeltaSpikeClassificationPort(ABC):
    """Contrato para clasificación de spikes (delta vs noise)."""

    @abstractmethod
    def classify(
        self,
        values: List[float],
        spike_index: int,
    ) -> DeltaSpikeResult:
        """Clasifica un spike detectado.

        Args:
            values: Serie temporal completa.
            spike_index: Índice donde ocurre el spike.

        Returns:
            ``DeltaSpikeResult`` con clasificación y razones.
        """
        ...


class RegimeDetectionPort(ABC):
    """Contrato para detección de regímenes operacionales."""

    @abstractmethod
    def train(self, historical_values: List[float]) -> None:
        """Entrena detector con datos históricos.

        Args:
            historical_values: Serie temporal de entrenamiento.
        """
        ...

    @abstractmethod
    def predict_regime(self, value: float) -> OperationalRegime:
        """Predice régimen para un valor dado.

        Args:
            value: Valor actual del sensor.

        Returns:
            ``OperationalRegime`` más probable.
        """
        ...

    @abstractmethod
    def is_trained(self) -> bool:
        """``True`` si el detector fue entrenado."""
        ...
