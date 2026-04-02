"""Value Objects para lecturas de sensores.

Inmutables, sin lógica de infraestructura.  Representan el dato crudo
que entra al sistema y la ventana temporal sobre la que operan los motores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True, slots=True)
class Reading:
    """UTSAE Canonical Reading type - unified sensor reading with series_id.
    
    Phase 6: Replaces SensorReading with series_id (str) as universal key.
    Eliminates sensor_id: int vs series_id: str duality.
    
    Attributes:
        series_id: Universal series identifier (string).
        value: Numeric reading value.
        timestamp: Unix timestamp.
        sensor_type: Type classification.
        device_id: Optional device reference.
    """
    
    series_id: str
    value: float
    timestamp: float
    sensor_type: str = ""
    device_id: Optional[int] = None
    
    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"Reading.value must be finite, got {self.value}")
        if not math.isfinite(self.timestamp):
            raise ValueError(f"Reading.timestamp must be finite, got {self.timestamp}")
    
    @property
    def is_valid(self) -> bool:
        return math.isfinite(self.value) and self.timestamp > 0
    
    # GOLD: @deprecated - Use series_id instead
    # Backward compatibility alias only - will be removed in v0.3.0
    @property
    def sensor_id(self) -> int:
        """@deprecated Use series_id (str) instead. Maintained for backward compatibility only."""
        try:
            return int(self.series_id)
        except ValueError:
            return 0


@dataclass(frozen=True, slots=True)
class SensorReading(Reading):
    """@deprecated Legacy alias for Reading. Use Reading directly."""
    pass


@dataclass(frozen=True, slots=True)
class TimeSeriesWindow:
    """UTSAE Canonical window - unified temporal window with series_id.
    
    Phase 6: Replaces SensorWindow with series_id (str) as universal key.
    
    Attributes:
        series_id: Universal series identifier (string).
        readings: Ordered chronologically.
        sensor_type: Type classification.
        device_id: Optional device reference.
    """
    
    series_id: str
    readings: List[Reading] = field(default_factory=list)
    sensor_type: str = ""
    device_id: Optional[int] = None
    
    @property
    def values(self) -> List[float]:
        return [r.value for r in self.readings]
    
    @property
    def timestamps(self) -> List[float]:
        return [r.timestamp for r in self.readings]
    
    @property
    def size(self) -> int:
        return len(self.readings)
    
    @property
    def is_empty(self) -> bool:
        return len(self.readings) == 0
    
    @property
    def last_value(self) -> Optional[float]:
        return self.readings[-1].value if self.readings else None
    
    @property
    def last_timestamp(self) -> Optional[float]:
        return self.readings[-1].timestamp if self.readings else None
    
    # GOLD: @deprecated - Use series_id instead
    @property
    def sensor_id(self) -> int:
        """@deprecated Use series_id (str) instead. Maintained for backward compatibility only."""
        try:
            return int(self.series_id)
        except ValueError:
            return 0


@dataclass(frozen=True, slots=True)
class SensorWindow(TimeSeriesWindow):
    """@deprecated Legacy alias for TimeSeriesWindow. Use TimeSeriesWindow directly."""
    pass

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

    @property
    def temporal_diagnostic(self) -> "TemporalDiagnostic":
        """Diagnóstico de calidad temporal de la ventana.

        Returns:
            ``TemporalDiagnostic`` con métricas de monotonía, gaps y duplicados.
        """
        from ...validators.temporal import diagnose_temporal_quality

        return diagnose_temporal_quality(self.timestamps)

    @property
    def temporal_features(self) -> "TemporalFeatures":
        """Features dinámicas derivadas: velocidad, aceleración, jitter.

        Delega a ``compute_temporal_features`` sobre valores y timestamps.

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

    def to_time_series(self) -> "TimeSeries":
        """Convierte a TimeSeries agnóstica (Nivel 1 UTSAE).

        Returns:
            ``TimeSeries`` con ``series_id = str(sensor_id)``.
        """
        from ..series.time_series import TimeSeries

        return TimeSeries.from_values(
            values=self.values,
            timestamps=self.timestamps,
            series_id=str(self.sensor_id),
        )
