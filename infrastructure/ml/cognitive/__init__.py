"""Cognitive orchestration layer for UTSAE.

Package structure:
    types.py                — EnginePerception, MetaDiagnostic, SignalProfile (deprecated)
    signal_analyzer.py      — noise/regime/stability analysis from raw signal
    inhibition.py           — weight suppression for unstable engines
    bayesian_weight_tracker — regime-contextual weight learning (renamed from 'plasticity')
    engine_selector.py      — weighted fusion + engine ranking
    orchestration/          — MetaCognitiveOrchestrator (modularized)
        orchestrator.py      — Core orchestrator class
        pipeline_executor.py — Pipeline execution logic
        fallback_handler.py  — Fallback handling

.. versionchanged:: 2.0
    ``SignalAnalyzer.analyze()`` now returns ``StructuralAnalysis`` (domain)
    instead of ``SignalProfile`` (infra).  ``SignalProfile`` is deprecated.

.. versionchanged:: 2.1
    ``orchestrator.py`` modularized into ``orchestration/`` subpackage.
    Backward compatibility maintained via re-exports.

.. versionchanged:: 2.2
    ``plasticity`` renamed to ``bayesian_weight_tracker`` for honest naming.
    Uses Bayesian inference — NOT reinforcement learning.
"""

from __future__ import annotations

from .fusion import WeightedFusion
from .explanation import ExplanationBuilder
from .inhibition import InhibitionGate

# Orchestration subpackage (backward compat for meta-tests)
from . import orchestration as orchestrator

try:
    from .orchestration import MetaCognitiveOrchestrator
except (ImportError, ModuleNotFoundError):
    MetaCognitiveOrchestrator = None  # type: ignore[assignment,misc]

try:
    from .bayesian_weight_tracker import BayesianWeightTracker, PlasticityTracker
except (ImportError, ModuleNotFoundError):
    BayesianWeightTracker = None  # type: ignore[assignment,misc]
    PlasticityTracker = None  # type: ignore[assignment,misc]

try:
    from .severity_classifier import SeverityClassifier
except (ImportError, ModuleNotFoundError):
    SeverityClassifier = None  # type: ignore[assignment,misc]

try:
    from .analysis import SignalAnalyzer
except (ImportError, ModuleNotFoundError):
    SignalAnalyzer = None  # type: ignore[assignment,misc]
from .analysis.types import (
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
    "SeverityClassifier",
    "SignalAnalyzer",
    "SignalProfile",
    "WeightedFusion",
]
