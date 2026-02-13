"""Entidades y Value Objects del dominio UTSAE.

Organización por subcarpetas:
- ``series/``   — Primitivos de series temporales (TimeSeries, TemporalFeatures, StructuralAnalysis, SeriesProfile, SeriesContext)
- ``patterns/`` — Resultados de detección de patrones (PatternResult, ChangePoint, DeltaSpikeResult, OperationalRegime)
- ``results/``  — Resultados de inferencia (AnomalyResult, Prediction, MemorySearchResult)
- ``iot/``      — Entidades IoT legacy (SensorReading, SensorWindow, sensor_ranges)

Todos los imports legacy siguen funcionando gracias a facades de re-export
en los archivos raíz (e.g. ``from .anomaly import ...``).
"""

# --- IoT (legacy boundary) ---
from .iot.sensor_reading import SensorReading, SensorWindow

# --- Results (inference output) ---
from .results.prediction import Prediction, PredictionConfidence
from .results.anomaly import AnomalyResult, AnomalySeverity
from .results.memory_search_result import MemorySearchResult

# --- Patterns ---
from .patterns.pattern_result import PatternResult, PatternType
from .patterns.change_point import ChangePoint, ChangePointType
from .patterns.delta_spike import DeltaSpikeResult, SpikeClassification
from .patterns.operational_regime import OperationalRegime

# --- Series (mathematical primitives) ---
from .series.temporal_features import TemporalFeatures
from .series.structural_analysis import StructuralAnalysis, RegimeType

__all__ = [
    # IoT
    "SensorReading",
    "SensorWindow",
    # Results
    "Prediction",
    "PredictionConfidence",
    "AnomalyResult",
    "AnomalySeverity",
    "MemorySearchResult",
    # Patterns
    "PatternResult",
    "PatternType",
    "ChangePoint",
    "ChangePointType",
    "DeltaSpikeResult",
    "SpikeClassification",
    "OperationalRegime",
    # Series
    "TemporalFeatures",
    "StructuralAnalysis",
    "RegimeType",
]
