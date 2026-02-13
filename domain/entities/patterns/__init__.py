"""Entidades de detección de patrones de comportamiento.

Resultados de análisis de patrones, puntos de cambio, spikes y regímenes.
"""

from .pattern_result import PatternType, PatternResult
from .change_point import ChangePointType, ChangePoint
from .delta_spike import SpikeClassification, DeltaSpikeResult
from .operational_regime import OperationalRegime

__all__ = [
    "PatternType",
    "PatternResult",
    "ChangePointType",
    "ChangePoint",
    "SpikeClassification",
    "DeltaSpikeResult",
    "OperationalRegime",
]
