"""Assembly Phase — GOLD version 0.2.1.

Constructs final PredictionResult from accumulated context.
GOLD: Added cognitive_trace combining drift, shadow, and circuit breaker status.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from ....interfaces import PredictionResult
from ...compliance import ComplianceExporter
from ...compliance.compliance_exporter import load_hmac_key_from_env

logger = logging.getLogger(__name__)


# IMP-5: process-level lazy ComplianceExporter.
#
# Resolved the first time AssemblyPhase sees the ``ML_COMPLIANCE_EXPORT_PATH``
# env var. Subsequent requests reuse the same exporter so the sink file is
# opened+appended with a shared lock across concurrent pipelines.
_compliance_exporter: Optional[ComplianceExporter] = None
_compliance_exporter_lock = Lock()
_compliance_export_path: Optional[str] = None


def _resolve_compliance_exporter() -> Optional[ComplianceExporter]:
    """Return the process-level exporter, creating it on first use."""
    global _compliance_exporter, _compliance_export_path
    path = os.environ.get("ML_COMPLIANCE_EXPORT_PATH")
    if not path:
        return None
    with _compliance_exporter_lock:
        if _compliance_exporter is None or _compliance_export_path != path:
            try:
                _compliance_exporter = ComplianceExporter(
                    sink_path=Path(path),
                    hmac_key=load_hmac_key_from_env(),
                )
                _compliance_export_path = path
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "compliance_exporter_init_failed",
                    extra={"path": path, "error": str(exc)},
                )
                _compliance_exporter = None
    return _compliance_exporter


def _reset_compliance_exporter() -> None:
    """Test-only helper: clear the cached exporter."""
    global _compliance_exporter, _compliance_export_path
    with _compliance_exporter_lock:
        _compliance_exporter = None
        _compliance_export_path = None


@dataclass(frozen=True)
class CognitiveTrace:
    """GOLD: Unified cognitive trace combining all phase metadata.
    
    Attributes:
        drift_score: Concept drift detection score (0 = no drift, >2 = drift)
        shadow_performance: Shadow engine evaluation results if enabled
        circuit_breaker_status: "closed", "open", or "half_open"
        amnesic_mode: True if running in RAM-only mode (persistence failed)
        assembly_timestamp: Unix timestamp of assembly
    """
    drift_score: float = 0.0
    shadow_performance: Optional[Dict[str, Any]] = None
    circuit_breaker_status: str = "closed"
    amnesic_mode: bool = False
    assembly_timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_score": self.drift_score,
            "shadow_performance": self.shadow_performance,
            "circuit_breaker_status": self.circuit_breaker_status,
            "amnesic_mode": self.amnesic_mode,
            "assembly_timestamp": self.assembly_timestamp,
        }


class AssemblyPhase:
    """Final phase: Assemble metadata and create PredictionResult."""
    
    @property
    def name(self) -> str:
        return "assembly"
    
    def execute(self, ctx: PipelineContext) -> PredictionResult:
        """Assemble final result from context."""
        metadata = self._build_metadata(ctx)
        
        # Get confidence interval if storage available
        orchestrator = ctx.orchestrator
        if orchestrator._storage and ctx.series_id != "unknown":
            try:
                ci = orchestrator._storage.compute_confidence_interval(
                    ctx.series_id, ctx.selected_engine, ctx.fused_value
                )
                if ci:
                    metadata["confidence_interval"] = ci
            except Exception as e:
                logger.debug(f"confidence_interval_failed: {e}")
        
        result = PredictionResult(
            predicted_value=ctx.fused_value,
            confidence=ctx.fused_confidence,
            trend=ctx.fused_trend,
            metadata=metadata,
        )
        # IMP-5: opt-in audit export. Must NEVER corrupt the return
        # envelope — any export failure is logged and swallowed.
        self._maybe_export_compliance(ctx.series_id, result)
        return result
    
    @staticmethod
    def _maybe_export_compliance(series_id: str, result: PredictionResult) -> None:
        exporter = _resolve_compliance_exporter()
        if exporter is None:
            return
        try:
            exporter.export(series_id, result)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "compliance_export_hook_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )
    
    def _build_metadata(self, ctx: PipelineContext) -> Dict[str, Any]:
        """Build comprehensive metadata dict with GOLD cognitive trace."""
        metadata: Dict[str, Any] = {
            "cognitive_diagnostic": ctx.diagnostic.to_dict() if ctx.diagnostic else None,
            "explanation": ctx.explanation.to_dict() if ctx.explanation else None,
            "pipeline_timing": ctx.timer.to_dict() if ctx.timer else None,
            "cognitive_trace": self._build_cognitive_trace(ctx),
            # IMP-1: always surface sanitization flags (empty list on clean input).
            "sanitization_flags": list(getattr(ctx, "sanitization_flags", [])),
            # IMP-2: fusion/Hampel flags + per-engine failure visibility.
            "fusion_flags": list(getattr(ctx, "fusion_flags", [])),
            "engine_failures": list(getattr(ctx, "engine_failures", [])),
        }
        # IMP-2: include Hampel diagnostic when something was computed.
        hampel_diag = getattr(ctx, "hampel_diagnostic", None)
        if hampel_diag:
            metadata["hampel"] = hampel_diag
        
        # Add boundary check if available
        if ctx.boundary_result is not None and ctx.boundary_result.within_domain:
            metadata["boundary_check"] = {
                "within_domain": True,
                "data_quality_score": ctx.boundary_result.data_quality_score,
                "warnings": ctx.boundary_result.warnings,
            }
        
        # Add coherence check
        if ctx.coherence_result is not None:
            metadata["coherence_check"] = {
                "is_coherent": ctx.coherence_result.is_coherent,
                "conflict_type": ctx.coherence_result.conflict_type,
                "resolved_confidence": ctx.coherence_result.resolved_confidence,
                "resolution_reason": ctx.coherence_result.resolution_reason,
            }
        
        # Add engine decision
        if ctx.engine_decision is not None:
            metadata["engine_decision"] = {
                "chosen_engine": ctx.engine_decision.chosen_engine,
                "authority": ctx.engine_decision.authority,
                "reason": ctx.engine_decision.reason,
                "overrides": ctx.engine_decision.overrides,
            }
        
        # Add calibration report
        if ctx.calibrated_confidence is not None:
            metadata["calibration_report"] = {
                "raw": ctx.calibrated_confidence.raw,
                "calibrated": ctx.calibrated_confidence.calibrated,
                "penalty_applied": ctx.calibrated_confidence.penalty_applied,
                "reasons": ctx.calibrated_confidence.reasons,
            }
        
        # Add action guard
        if ctx.guarded_action is not None:
            metadata["action_guard"] = {
                "action_allowed": ctx.guarded_action.action_allowed,
                "original_action": ctx.guarded_action.original_action,
                "final_action": ctx.guarded_action.final_action,
                "suppressed_reason": ctx.guarded_action.suppressed_reason,
                "series_state": ctx.guarded_action.series_state,
            }
        
        # Add unified narrative
        if ctx.unified_narrative is not None:
            metadata["unified_narrative"] = {
                "primary_verdict": ctx.unified_narrative.primary_verdict,
                "severity": ctx.unified_narrative.severity,
                "confidence": ctx.unified_narrative.confidence,
                "contradictions": ctx.unified_narrative.contradictions,
                "sources_used": ctx.unified_narrative.sources_used,
                "suppressed": ctx.unified_narrative.suppressed,
            }
        
        return metadata
    
    def _build_cognitive_trace(self, ctx: PipelineContext) -> Dict[str, Any]:
        """GOLD: Build unified cognitive trace from all phase outputs."""
        trace = CognitiveTrace(assembly_timestamp=time.time())
        
        # Extract drift information
        if hasattr(ctx, 'drift_response') and ctx.drift_response:
            trace = CognitiveTrace(
                drift_score=ctx.drift_response.get('drift_score', 0.0),
                amnesic_mode=ctx.drift_response.get('amnesic_mode', False),
                assembly_timestamp=trace.assembly_timestamp,
            )
        
        # Extract shadow performance
        if hasattr(ctx, 'experimental_metadata') and ctx.experimental_metadata:
            shadow = ctx.experimental_metadata.get('shadow')
            if shadow and isinstance(shadow, dict):
                trace = CognitiveTrace(
                    drift_score=trace.drift_score,
                    shadow_performance={
                        "engines_tested": shadow.get('engines_tested', 0),
                        "results": shadow.get('results', []),
                        "sampled": shadow.get('sampled', True),
                    },
                    circuit_breaker_status=trace.circuit_breaker_status,
                    amnesic_mode=trace.amnesic_mode,
                    assembly_timestamp=trace.assembly_timestamp,
                )
        
        # Extract circuit breaker status from persistence adapter
        if hasattr(ctx, 'persistence_adapter') and ctx.persistence_adapter:
            adapter = ctx.persistence_adapter
            if hasattr(adapter, '_circuit'):
                circuit = adapter._circuit
                if circuit.is_open:
                    status = "open"
                elif circuit.failures > 0:
                    status = "half_open"
                else:
                    status = "closed"
                trace = CognitiveTrace(
                    drift_score=trace.drift_score,
                    shadow_performance=trace.shadow_performance,
                    circuit_breaker_status=status,
                    amnesic_mode=trace.amnesic_mode or circuit.is_open,
                    assembly_timestamp=trace.assembly_timestamp,
                )
        
        return trace.to_dict()
