"""Primitivos de series temporales — Nivel 1 (Matemático).

Contiene las entidades fundamentales que UTSAE percibe:
secuencias numéricas con tiempo, features derivadas y análisis estructural.
"""

from .time_series import TimeSeries, TimePoint
from .temporal_features import TemporalFeatures
from .structural_analysis import StructuralAnalysis, RegimeType
from .series_profile import (
    SeriesProfile,
    VolatilityLevel,
    StationarityHint,
    compute_profile,
)
from .series_context import SeriesContext, Threshold

# Import from parent canonical_series.py file (Zenin canonical types)
from ..canonical_series import DataPoint, TimeWindow

__all__ = [
    "TimeSeries",
    "TimePoint",
    "TemporalFeatures",
    "StructuralAnalysis",
    "RegimeType",
    "SeriesProfile",
    "VolatilityLevel",
    "StationarityHint",
    "compute_profile",
    "SeriesContext",
    "Threshold",
    "DataPoint",
    "TimeWindow",
]
