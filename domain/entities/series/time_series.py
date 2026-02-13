"""Representación agnóstica de una serie temporal — Nivel 1 (Matemático).

UTSAE no sabe si esto es temperatura, ventas, latencia o emociones.
Solo ve: secuencia numérica con tiempo.

TimeSeries es el primitivo fundamental del motor cognitivo.
Todo análisis de Nivel 1 opera sobre esta estructura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TimePoint:
    """Un punto en el tiempo con un valor numérico.

    Attributes:
        t: Timestamp (unidades arbitrarias — segundos, días, índice).
        v: Valor numérico observado.
    """

    t: float
    v: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError(f"TimePoint.v debe ser finito, recibido {self.v}")
        if not math.isfinite(self.t):
            raise ValueError(f"TimePoint.t debe ser finito, recibido {self.t}")


@dataclass(frozen=True)
class TimeSeries:
    """Secuencia temporal ordenada — el dato crudo que UTSAE percibe.

    Invariantes:
    - Al menos 1 punto.
    - Ordenada cronológicamente (t creciente).
    - Todos los valores finitos.

    Attributes:
        series_id: Identificador opaco de la serie (str para ser agnóstico).
        points: Puntos ordenados cronológicamente.
    """

    series_id: str
    points: List[TimePoint] = field(default_factory=list)

    @property
    def values(self) -> List[float]:
        """Valores numéricos en orden cronológico."""
        return [p.v for p in self.points]

    @property
    def timestamps(self) -> List[float]:
        """Timestamps en orden cronológico."""
        return [p.t for p in self.points]

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0

    @property
    def duration(self) -> float:
        """Duración total de la serie en unidades de tiempo."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].t - self.points[0].t

    @property
    def last(self) -> Optional[TimePoint]:
        """Último punto, o None si vacía."""
        return self.points[-1] if self.points else None

    @property
    def first(self) -> Optional[TimePoint]:
        """Primer punto, o None si vacía."""
        return self.points[0] if self.points else None

    @property
    def mean_dt(self) -> float:
        """Intervalo medio entre puntos consecutivos."""
        if len(self.points) < 2:
            return 1.0
        return self.duration / (len(self.points) - 1)

    @property
    def temporal_features(self) -> "TemporalFeatures":
        """Features dinámicas derivadas: velocidad, aceleración, jitter.

        Returns:
            ``TemporalFeatures`` con métricas temporales calculadas.
        """
        from ...validators.temporal_features import compute_temporal_features

        return compute_temporal_features(self.values, self.timestamps)

    @property
    def structural_analysis(self) -> "StructuralAnalysis":
        """Análisis estructural unificado: slope, curvature, stability, régimen.

        Returns:
            ``StructuralAnalysis`` con métricas estructurales calculadas.
        """
        from ...validators.structural_analysis import compute_structural_analysis

        return compute_structural_analysis(self.values, self.timestamps)

    @classmethod
    def from_values(
        cls,
        values: List[float],
        timestamps: Optional[List[float]] = None,
        series_id: str = "anonymous",
    ) -> TimeSeries:
        """Crea TimeSeries desde listas planas.

        Si no se proveen timestamps, se asume muestreo uniforme (dt=1).

        Args:
            values: Valores numéricos.
            timestamps: Timestamps opcionales (misma longitud que values).
            series_id: Identificador de la serie.

        Returns:
            TimeSeries construida.
        """
        if timestamps is None:
            timestamps = [float(i) for i in range(len(values))]

        if len(values) != len(timestamps):
            raise ValueError(
                f"values ({len(values)}) y timestamps ({len(timestamps)}) "
                f"deben tener la misma longitud"
            )

        points = [TimePoint(t=t, v=v) for t, v in zip(timestamps, values)]
        return cls(series_id=series_id, points=points)

    @classmethod
    def from_sensor_window(cls, window: object) -> TimeSeries:
        """Bridge: convierte SensorWindow (IoT) a TimeSeries (agnóstico).

        Args:
            window: Instancia de ``SensorWindow`` (import diferido para
                no acoplar este módulo a IoT).

        Returns:
            ``TimeSeries`` con ``series_id = str(window.sensor_id)``.
        """
        return cls.from_values(
            values=window.values,  # type: ignore[attr-defined]
            timestamps=window.timestamps,  # type: ignore[attr-defined]
            series_id=str(window.sensor_id),  # type: ignore[attr-defined]
        )
