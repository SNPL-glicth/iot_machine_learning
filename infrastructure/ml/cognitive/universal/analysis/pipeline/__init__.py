"""Universal analysis pipeline phases.

Modular pipeline components for input-agnostic cognitive analysis.
"""

from .perceive_phase import PerceivePhase
from .analyze_phase import AnalyzePhase
from .remember_phase import RememberPhase
from .reason_phase import ReasonPhase
from .explain_phase import ExplainPhase

__all__ = [
    "PerceivePhase",
    "AnalyzePhase", 
    "RememberPhase",
    "ReasonPhase",
    "ExplainPhase",
]
