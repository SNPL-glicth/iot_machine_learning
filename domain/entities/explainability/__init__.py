"""Capa de explicabilidad cognitiva — value objects domain-pure.

Estructura:
- ``explanation.py``            — Explanation (raíz), Outcome
- ``reasoning_trace.py``        — ReasoningTrace, ReasoningPhase, PhaseKind
- ``contribution_breakdown.py`` — ContributionBreakdown, EngineContribution
- ``signal_snapshot.py``        — SignalSnapshot, FilterSnapshot
"""

from .explanation import Explanation, Outcome
from .reasoning_trace import PhaseKind, ReasoningPhase, ReasoningTrace
from .contribution_breakdown import ContributionBreakdown, EngineContribution
from .signal_snapshot import FilterSnapshot, SignalSnapshot

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
]
