"""Value Objects para lecturas de sensores.

Inmutables, sin lógica de infraestructura.  Representan el dato crudo
que entra al sistema y la ventana temporal sobre la que operan los motores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class SensorReading:
    """Lectura individual de un sensor IoT.

    Attributes:
        sensor_id: Identificador único del sensor.
        value: Valor numérico de la lectura.
        timestamp: Timestamp Unix (segundos desde epoch).
        sensor_type: Tipo de sensor (``"temperature"``, ``"humidity"``, etc.).
        device_id: ID del dispositivo al que pertenece el sensor.
    """

    sensor_id: int
    value: float
    timestamp: float
    sensor_type: str = ""
    device_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(
                f"SensorReading.value debe ser finito, recibido {self.value}"
            )

    @property
    def is_valid(self) -> bool:
        """``True`` si el valor es finito y el timestamp es positivo."""
        return math.isfinite(self.value) and self.timestamp > 0


@dataclass(frozen=True)
class SensorWindow:
    """Ventana temporal de lecturas de un sensor.

    Contiene N lecturas ordenadas cronológicamente (más antiguo primero).
    Provee acceso conveniente a valores y timestamps como listas.

    Attributes:
        sensor_id: ID del sensor.
        readings: Lecturas ordenadas cronológicamente.
        sensor_type: Tipo de sensor.
        device_id: ID del dispositivo.
    """

    sensor_id: int
    readings: List[SensorReading] = field(default_factory=list)
    sensor_type: str = ""
    device_id: Optional[int] = None

    @property
    def values(self) -> List[float]:
        """Lista de valores numéricos."""
        return [r.value for r in self.readings]

    @property
    def timestamps(self) -> List[float]:
        """Lista de timestamps Unix."""
        return [r.timestamp for r in self.readings]

    @property
    def size(self) -> int:
        """Número de lecturas en la ventana."""
        return len(self.readings)

    @property
    def is_empty(self) -> bool:
        """``True`` si no hay lecturas."""
        return len(self.readings) == 0

    @property
    def last_value(self) -> Optional[float]:
        """Último valor de la ventana, o ``None`` si está vacía."""
        if self.readings:
            return self.readings[-1].value
        return None

    @property
    def last_timestamp(self) -> Optional[float]:
        """Último timestamp, o ``None`` si está vacía."""
        if self.readings:
            return self.readings[-1].timestamp
        return None

    @property
    def time_span_seconds(self) -> float:
        """Duración total de la ventana en segundos."""
        if len(self.readings) < 2:
            return 0.0
        return self.readings[-1].timestamp - self.readings[0].timestamp

    def to_time_series(self) -> "TimeSeries":
        """Convierte a TimeSeries agnóstica (Nivel 1 UTSAE).

        Returns:
            ``TimeSeries`` con ``series_id = str(sensor_id)``.
        """
        from .time_series import TimeSeries

        return TimeSeries.from_values(
            values=self.values,
            timestamps=self.timestamps,
            series_id=str(self.sensor_id),
        )
