"""Port de detección de anomalías — contrato para detectores.

Desacopla el dominio de implementaciones concretas (IsolationForest,
LOF, Z-score, Voting Ensemble, etc.).

Dual interface: acepta ``SensorWindow`` (legacy) o ``TimeSeries`` (agnóstico).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.anomaly import AnomalyResult
from ..entities.sensor_reading import SensorWindow
from ..validators.input_guard import safe_series_id_to_int
from ..entities.time_series import TimeSeries


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
    def train(
        self,
        historical_values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """Entrena el detector con datos históricos.

        Args:
            historical_values: Serie temporal de entrenamiento.
            timestamps: Timestamps correspondientes (opcional).
                Si se proveen, habilita features temporales
                (velocidad, aceleración) en detectores que lo soporten.

        Raises:
            ValueError: Si no hay suficientes datos.
        """
        ...

    @abstractmethod
    def detect(self, window: SensorWindow) -> AnomalyResult:
        """Detecta anomalías en la ventana actual (legacy).

        Args:
            window: Ventana temporal del sensor.

        Returns:
            ``AnomalyResult`` con score, votos, explicación.
        """
        ...

    def detect_series(self, series: TimeSeries) -> AnomalyResult:
        """Detecta anomalías en una TimeSeries agnóstica.

        Implementación por defecto delega a ``detect`` vía bridge.
        """
        from ..entities.sensor_reading import SensorReading

        readings = [
            SensorReading(
                sensor_id=safe_series_id_to_int(series.series_id),
                value=p.v,
                timestamp=p.t,
            )
            for p in series.points
        ]
        sw = SensorWindow(
            sensor_id=safe_series_id_to_int(series.series_id),
            readings=readings,
        )
        return self.detect(sw)

    @abstractmethod
    def is_trained(self) -> bool:
        """``True`` si el detector fue entrenado."""
        ...
