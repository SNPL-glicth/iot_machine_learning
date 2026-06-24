"""Pipeline Context — MED-1 Refactoring.

Immutable context object that flows through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineContext:
    """Mutable context that flows through the pipeline.

    Each phase receives the same context and writes to it
    directly.  No copies are made between phases, eliminating
    the O(n) object‑explosion overhead of the previous
    immutable design.
    """
    
    # Input (required)
    orchestrator: Any
    values: List[float]
    timestamps: Optional[List[float]]
    series_id: str
    flags: Any
    timer: Any
    
    # Phase outputs (initialized to None, filled progressively)
    # IMP-1: sanitize phase outputs.
    sanitized_values: Optional[List[float]] = None
    sanitization_flags: List[str] = field(default_factory=list)
    # IMP-2: fusion/Hampel outputs + per-engine failure surface.
    fusion_flags: List[str] = field(default_factory=list)
    hampel_diagnostic: Optional[Dict[str, Any]] = None
    engine_failures: List[Dict[str, Any]] = field(default_factory=list)
    boundary_result: Optional[Any] = None
    profile: Optional[Any] = None
    regime: Optional[str] = None
    neighbor_trends: Optional[Dict[str, str]] = None
    neighbors: Optional[List] = None
    neighbor_values: Optional[Dict] = None
    plasticity_context: Optional[Any] = None
    feature_context: Optional[Any] = None
    perceptions: Optional[List] = None
    error_dict: Optional[Dict[str, List[float]]] = None
    plasticity_weights: Optional[Dict[str, float]] = None
    inhibition_states: Optional[List] = None
    mediated_weights: Optional[Dict[str, float]] = None
    fused_value: Optional[float] = None
    fused_confidence: Optional[float] = None
    raw_fused_confidence: Optional[float] = None  # Preserved before calibration
    fused_trend: Optional[str] = None
    final_weights: Optional[Dict[str, float]] = None
    selected_engine: Optional[str] = None
    selection_reason: Optional[str] = None
    fusion_method: Optional[str] = None
    engine_decision: Optional[Any] = None
    coherence_result: Optional[Any] = None
    calibrated_confidence: Optional[Any] = None
    guarded_action: Optional[Any] = None
    unified_narrative: Optional[Any] = None
    diagnostic: Optional[Any] = None
    explanation: Optional[Any] = None
    explanation_summary: Optional[str] = None
    consecutive_anomalies: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    
    # PredictionReadinessGate output
    data_quality_score: float = 1.0
    max_action: str = "PREDICT"

    # Regime enrichment from PerceivePhase
    regime_confidence: float = 0.5
    cross_regime_incoherence: bool = False

    # Drift enrichment from DriftDetectionPhase
    drift_detected: bool = False
    drift_magnitude: float = 0.0
    drift_type: Optional[str] = None
    drift_cause: Optional[str] = None
    condition_indicator: Optional[Dict[str, Any]] = None

    # Causal chain from CausalPhase
    causal_events: List[Dict[str, Any]] = field(default_factory=list)

    # Observability components (Phase 3C)
    metrics_collector: Optional[Any] = None
    memory_health_monitor: Optional[Any] = None
    
    # Memory components (Phase 3A)
    memory_registry: Optional[Any] = None
    
    def with_field(self, **kwargs) -> PipelineContext:
        """Update fields in place and return self for backward compatibility."""
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)
        return self


def create_initial_context(
    orchestrator,
    values: List[float],
    timestamps: Optional[List[float]],
    series_id: str,
    flags: Any,
    timer: Any,
) -> PipelineContext:
    """Factory function to create initial pipeline context."""
    metrics_collector = getattr(orchestrator, '_metrics_collector', None)
    memory_health_monitor = getattr(orchestrator, '_memory_health_monitor', None)
    memory_registry = getattr(orchestrator, '_memory_registry', None)
    return PipelineContext(
        orchestrator=orchestrator,
        values=values,
        timestamps=timestamps,
        series_id=series_id,
        flags=flags,
        timer=timer,
        metrics_collector=metrics_collector,
        memory_health_monitor=memory_health_monitor,
        memory_registry=memory_registry,
    )
