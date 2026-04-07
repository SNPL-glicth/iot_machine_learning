"""Pipeline modular de análisis — 4 fases."""

from .perceive import PerceivePhase
from .analyze import AnalyzePhase
from .reason import ReasonPhase
from .explain import ExplainPhase

__all__ = [
    "PerceivePhase",
    "AnalyzePhase",
    "ReasonPhase",
    "ExplainPhase",
]
