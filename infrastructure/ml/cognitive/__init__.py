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

.. versionchanged:: 2.3
    Added observability components (Phase 3C) for cognitive hardening.
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

# Observability components (Phase 3C)
try:
    from .observability import (
        CognitiveMetricsCollector,
        MemoryHealthMonitor,
        DriftDetectionEngine,
        ExplainabilityValidator,
        FeedbackLoopManager,
        CognitiveObservabilityDashboard,
    )
except (ImportError, ModuleNotFoundError):
    CognitiveMetricsCollector = None  # type: ignore[assignment,misc]
    MemoryHealthMonitor = None  # type: ignore[assignment,misc]
    DriftDetectionEngine = None  # type: ignore[assignment,misc]
    ExplainabilityValidator = None  # type: ignore[assignment,misc]
    FeedbackLoopManager = None  # type: ignore[assignment,misc]
    CognitiveObservabilityDashboard = None  # type: ignore[assignment,misc]

# Memory components (Phase 3A)
try:
    from .memory import (
        SemanticEventBuilder,
        AnomalyMemoryStore,
        OperationalMemoryPipeline,
        HistoricalSimilarityRetriever,
        CognitiveMemoryRegistry,
    )
except (ImportError, ModuleNotFoundError):
    SemanticEventBuilder = None  # type: ignore[assignment,misc]
    AnomalyMemoryStore = None  # type: ignore[assignment,misc]
    OperationalMemoryPipeline = None  # type: ignore[assignment,misc]
    HistoricalSimilarityRetriever = None  # type: ignore[assignment,misc]
    CognitiveMemoryRegistry = None  # type: ignore[assignment,misc]

# Explainability components (Phase 3B)
try:
    from .explainability import (
        ContextualExplainabilityEngine,
        HistoricalContextAggregator,
        RecommendationGenerator,
        ContextualConfidenceCalculator,
        OperationalSummaryBuilder,
    )
except (ImportError, ModuleNotFoundError):
    ContextualExplainabilityEngine = None  # type: ignore[assignment,misc]
    HistoricalContextAggregator = None  # type: ignore[assignment,misc]
    RecommendationGenerator = None  # type: ignore[assignment,misc]
    ContextualConfidenceCalculator = None  # type: ignore[assignment,misc]
    OperationalSummaryBuilder = None  # type: ignore[assignment,misc]

# Causal components (Phase 4A)
try:
    from .causal import (
        CausalCorrelationEngine,
        OperationalDependencyGraphManager,
        TemporalPatternMiner,
        EventPropagationTracker,
        PropagationConfidenceCalculator,
        OperationalSequenceRegistry,
    )
except (ImportError, ModuleNotFoundError):
    CausalCorrelationEngine = None  # type: ignore[assignment,misc]
    OperationalDependencyGraphManager = None  # type: ignore[assignment,misc]
    TemporalPatternMiner = None  # type: ignore[assignment,misc]
    EventPropagationTracker = None  # type: ignore[assignment,misc]
    PropagationConfidenceCalculator = None  # type: ignore[assignment,misc]
    OperationalSequenceRegistry = None  # type: ignore[assignment,misc]

# Neural components (High impact)
try:
    from .neural import HybridNeuralEngine, NeuralResult
except (ImportError, ModuleNotFoundError):
    HybridNeuralEngine = None  # type: ignore[assignment,misc]
    NeuralResult = None  # type: ignore[assignment,misc]

# Decision components (High impact)
try:
    from .decision import ContextualDecisionEngine
except (ImportError, ModuleNotFoundError):
    ContextualDecisionEngine = None  # type: ignore[assignment,misc]

# Pattern interpreter components (Medium-high impact)
try:
    from .universal.analysis.pattern_interpreter import PatternInterpreter, InterpretedPattern
except (ImportError, ModuleNotFoundError):
    PatternInterpreter = None  # type: ignore[assignment,misc]
    InterpretedPattern = None  # type: ignore[assignment,misc]

# Dynamic components (Medium-high impact)
try:
    from .dynamic import RollingWindowEngine, DynamicFeaturePipeline
except (ImportError, ModuleNotFoundError):
    RollingWindowEngine = None  # type: ignore[assignment,misc]
    DynamicFeaturePipeline = None  # type: ignore[assignment,misc]

# Regime detection components (Medium-high impact)
try:
    from .regime import RegimeDetectionPipeline, OperationalRegimeClassifier
except (ImportError, ModuleNotFoundError):
    RegimeDetectionPipeline = None  # type: ignore[assignment,misc]
    OperationalRegimeClassifier = None  # type: ignore[assignment,misc]

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
    # Observability components (Phase 3C)
    "CognitiveMetricsCollector",
    "MemoryHealthMonitor",
    "DriftDetectionEngine",
    "ExplainabilityValidator",
    "FeedbackLoopManager",
    "CognitiveObservabilityDashboard",
    # Memory components (Phase 3A)
    "SemanticEventBuilder",
    "AnomalyMemoryStore",
    "OperationalMemoryPipeline",
    "HistoricalSimilarityRetriever",
    "CognitiveMemoryRegistry",
    # Explainability components (Phase 3B)
    "ContextualExplainabilityEngine",
    "HistoricalContextAggregator",
    "RecommendationGenerator",
    "ContextualConfidenceCalculator",
    "OperationalSummaryBuilder",
    # Causal components (Phase 4A)
    "CausalCorrelationEngine",
    "OperationalDependencyGraphManager",
    "TemporalPatternMiner",
    "EventPropagationTracker",
    "PropagationConfidenceCalculator",
    "OperationalSequenceRegistry",
    # Neural components (High impact)
    "HybridNeuralEngine",
    "NeuralResult",
    # Decision components (High impact)
    "ContextualDecisionEngine",
    # Pattern interpreter components (Medium-high impact)
    "PatternInterpreter",
    "InterpretedPattern",
    # Dynamic components (Medium-high impact)
    "RollingWindowEngine",
    "DynamicFeaturePipeline",
    # Regime detection components (Medium-high impact)
    "RegimeDetectionPipeline",
    "OperationalRegimeClassifier",
]
