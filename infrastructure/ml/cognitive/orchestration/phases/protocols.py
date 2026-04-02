"""Phase Protocols — MED-1 Refactoring.

Defines the PipelinePhase protocol and TypedDict structures.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, TypedDict


class PhaseMetadata(TypedDict, total=False):
    """Metadata produced by a single pipeline phase."""
    phase_name: str
    duration_ms: float
    skipped: bool
    skip_reason: str | None


class PipelineResultMetadata(TypedDict, total=False):
    """Complete metadata structure for pipeline results."""
    cognitive_diagnostic: Dict[str, Any]
    explanation: Dict[str, Any]
    pipeline_timing: Dict[str, Any]
    boundary_check: Dict[str, Any]
    coherence_check: Dict[str, Any]
    engine_decision: Dict[str, Any]
    calibration_report: Dict[str, Any]
    action_guard: Dict[str, Any]
    unified_narrative: Dict[str, Any]
    confidence_interval: Dict[str, Any]


class PipelinePhase(Protocol):
    """Protocol for all pipeline phases.
    
    Each phase receives the current context, performs its work,
    and returns an updated context.
    """
    
    @property
    def name(self) -> str:
        """Phase name for logging and metrics."""
        ...
    
    def execute(self, ctx: "PipelineContext") -> "PipelineContext":
        """Execute the phase."""
        ...
