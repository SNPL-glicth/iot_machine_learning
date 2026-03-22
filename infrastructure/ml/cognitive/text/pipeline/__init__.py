"""Text cognitive pipeline phases.

Modular pipeline components for text cognitive analysis.
"""

from .perceive_phase import TextPerceivePhase
from .analyze_phase import TextAnalyzePhase
from .remember_phase import TextRememberPhase
from .reason_phase import TextReasonPhase
from .explain_phase import TextExplainPhase

__all__ = [
    "TextPerceivePhase",
    "TextAnalyzePhase", 
    "TextRememberPhase",
    "TextReasonPhase",
    "TextExplainPhase",
]
