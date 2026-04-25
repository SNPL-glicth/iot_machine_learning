"""Pipeline Context — MED-1 Refactoring.

Immutable context object that flows through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PipelineContext:
    """Immutable context that flows through the pipeline.
    
    Each phase receives a context and returns a new one with
    updated fields. This ensures data flow is explicit and
    prevents side effects between phases.
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
    perceptions: Optional[List] = None
    error_dict: Optional[Dict[str, List[float]]] = None
    plasticity_weights: Optional[Dict[str, float]] = None
    inhibition_states: Optional[List] = None
    mediated_weights: Optional[Dict[str, float]] = None
    fused_value: Optional[float] = None
    fused_confidence: Optional[float] = None
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    
    def with_field(self, **kwargs) -> PipelineContext:
        """Return new context with updated fields."""
        current_dict = {
            'orchestrator': self.orchestrator,
            'values': self.values,
            'timestamps': self.timestamps,
            'series_id': self.series_id,
            'flags': self.flags,
            'timer': self.timer,
            'sanitized_values': self.sanitized_values,
            'sanitization_flags': list(self.sanitization_flags),
            'fusion_flags': list(self.fusion_flags),
            'hampel_diagnostic': self.hampel_diagnostic,
            'engine_failures': list(self.engine_failures),
            'boundary_result': self.boundary_result,
            'profile': self.profile,
            'regime': self.regime,
            'neighbor_trends': self.neighbor_trends,
            'neighbors': self.neighbors,
            'neighbor_values': self.neighbor_values,
            'plasticity_context': self.plasticity_context,
            'perceptions': self.perceptions,
            'error_dict': self.error_dict,
            'plasticity_weights': self.plasticity_weights,
            'inhibition_states': self.inhibition_states,
            'mediated_weights': self.mediated_weights,
            'fused_value': self.fused_value,
            'fused_confidence': self.fused_confidence,
            'fused_trend': self.fused_trend,
            'final_weights': self.final_weights,
            'selected_engine': self.selected_engine,
            'selection_reason': self.selection_reason,
            'fusion_method': self.fusion_method,
            'engine_decision': self.engine_decision,
            'coherence_result': self.coherence_result,
            'calibrated_confidence': self.calibrated_confidence,
            'guarded_action': self.guarded_action,
            'unified_narrative': self.unified_narrative,
            'diagnostic': self.diagnostic,
            'explanation': self.explanation,
            'metadata': self.metadata.copy(),
            'is_fallback': self.is_fallback,
            'fallback_reason': self.fallback_reason,
        }
        current_dict.update(kwargs)
        return PipelineContext(**current_dict)


def create_initial_context(
    orchestrator,
    values: List[float],
    timestamps: Optional[List[float]],
    series_id: str,
    flags: Any,
    timer: Any,
) -> PipelineContext:
    """Factory function to create initial pipeline context."""
    return PipelineContext(
        orchestrator=orchestrator,
        values=values,
        timestamps=timestamps,
        series_id=series_id,
        flags=flags,
        timer=timer,
    )
