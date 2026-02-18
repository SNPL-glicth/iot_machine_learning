"""Cognitive orchestration layer for UTSAE.

Package structure:
    types.py           — EnginePerception, MetaDiagnostic, SignalProfile (deprecated)
    signal_analyzer.py — noise/regime/stability analysis from raw signal
    inhibition.py      — weight suppression for unstable engines
    plasticity.py      — regime-contextual weight learning
    engine_selector.py — weighted fusion + engine ranking
    orchestrator.py    — MetaCognitiveOrchestrator (top-level coordinator)

.. versionchanged:: 2.0
    ``SignalAnalyzer.analyze()`` now returns ``StructuralAnalysis`` (domain)
    instead of ``SignalProfile`` (infra).  ``SignalProfile`` is deprecated.
"""

from __future__ import annotations

from .engine_selector import WeightedFusion
from .builder import ExplanationBuilder
from .inhibition import InhibitionGate
from .orchestrator import MetaCognitiveOrchestrator
from .plasticity import PlasticityTracker
from .signal_analyzer import SignalAnalyzer
from .types import (
    EnginePerception,
    InhibitionState,
    MetaDiagnostic,
    PipelineTimer,
    SignalProfile,
)

__all__ = [
    "EnginePerception",
    "ExplanationBuilder",
    "InhibitionGate",
    "InhibitionState",
    "MetaCognitiveOrchestrator",
    "MetaDiagnostic",
    "PipelineTimer",
    "PlasticityTracker",
    "SignalAnalyzer",
    "SignalProfile",
    "WeightedFusion",
]
