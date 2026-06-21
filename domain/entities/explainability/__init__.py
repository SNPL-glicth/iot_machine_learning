"""Capa de explicabilidad cognitiva — value objects domain-pure.

Estructura:
- ``explanation.py``            — Explanation (raíz), Outcome
- ``reasoning_trace.py``        — ReasoningTrace, ReasoningPhase, PhaseKind
- ``contribution_breakdown.py`` — ContributionBreakdown, EngineContribution
- ``signal_snapshot.py``        — SignalSnapshot, FilterSnapshot
- ``contextual_explanation.py`` — ContextualExplanation (explainability con memoria operacional)
"""

from .explanation import Explanation, Outcome
from .reasoning_trace import PhaseKind, ReasoningPhase, ReasoningTrace
from .contribution_breakdown import ContributionBreakdown, EngineContribution
from .signal_snapshot import FilterSnapshot, SignalSnapshot
from .contextual_explanation import ContextualExplanation

__all__ = [
    "Explanation",
    "Outcome",
    "PhaseKind",
    "ReasoningPhase",
    "ReasoningTrace",
    "ContributionBreakdown",
    "EngineContribution",
    "FilterSnapshot",
    "SignalSnapshot",
    "ContextualExplanation",
]
