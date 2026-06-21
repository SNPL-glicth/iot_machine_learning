"""Pipeline Executor — MED-1 Refactored. Orquestador de fases cognitivas."""
from __future__ import annotations
import logging, os, time
from typing import TYPE_CHECKING, Any, List, Optional
if TYPE_CHECKING:
    from ...interfaces import PredictionResult
from .phases import PipelineContext, create_initial_context
from ..sanitize import SanitizePhase
from .phases.boundary_check_phase import BoundaryCheckPhase
from .phases.seasonal_decomposition_phase import SeasonalDecompositionPhase
from .phases.perceive_phase import PerceivePhase
from .phases.drift_detection_phase import DriftDetectionPhase
from .phases.predict_phase import PredictPhase
from .phases.adapt_phase import AdaptPhase
from .phases.inhibit_phase import InhibitPhase
from .phases.fuse_phase import FusePhase
from .phases.decision_arbiter_phase import DecisionArbiterPhase
from .phases.coherence_check_phase import CoherenceCheckPhase
from .phases.confidence_calibration_phase import ConfidenceCalibrationPhase
from .phases.explain_phase import ExplainPhase
from .phases.action_guard_phase import ActionGuardPhase
from .phases.narrative_unification_phase import NarrativeUnificationPhase
from .phases.assembly_phase import AssemblyPhase
from .phases.observability_phase import ObservabilityPhase
from .phases.memory_phase import MemoryPhase
from .phases.causal_phase import CausalPhase
from ..analysis.types import PipelineTimer
from ..compliance import ComplianceExporter
logger = logging.getLogger(__name__)

class PipelineExecutor:
    """Orquesta ejecución secuencial de fases."""
    def __init__(self, phases: Optional[list]=None, compliance_exporter: Optional[ComplianceExporter]=None) -> None:
        if phases is None:
            phases = [SanitizePhase(), BoundaryCheckPhase(), SeasonalDecompositionPhase(),
                      PerceivePhase(), DriftDetectionPhase(), PredictPhase(), AdaptPhase(),
                      InhibitPhase(), FusePhase(), DecisionArbiterPhase(), CoherenceCheckPhase(),
                      ConfidenceCalibrationPhase(), ExplainPhase(), ActionGuardPhase(),
                      NarrativeUnificationPhase(), MemoryPhase(), CausalPhase(),
                      ObservabilityPhase()]
        self._phases = phases
        self._assembly = AssemblyPhase(compliance_exporter=compliance_exporter)
    def execute(self, orchestrator, values: List[float], timestamps: Optional[List[float]],
                series_id: str, flags: Any) -> PredictionResult:
        if flags is None:
            raise ValueError("flags is required. Pass feature flags from orchestrator.")
        timer = PipelineTimer(budget_ms=orchestrator._budget_ms)
        ctx = create_initial_context(orchestrator=orchestrator, values=values, timestamps=timestamps,
                                     series_id=series_id, flags=flags, timer=timer)
        _warn_ms = int(os.environ.get("ML_COGNITIVE_PHASE_WARN_MS", "200"))
        _phase_times: dict[str, float] = {}
        _total_elapsed_ms = 0.0
        for phase in self._phases:
            phase_name = phase.__class__.__name__
            t0 = time.perf_counter()
            ctx = phase.execute(ctx)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            _phase_times[phase_name] = elapsed_ms
            _total_elapsed_ms += elapsed_ms
            if elapsed_ms > _warn_ms:
                logger.warning("[PROD-2] slow_phase %s %.1fms > %dms", phase_name, elapsed_ms, _warn_ms)
            try:
                from ...metrics.performance_metrics import record_cognitive_phase
                record_cognitive_phase(phase_name, elapsed_ms)
            except Exception:
                pass
            if ctx.is_fallback and ctx.fallback_reason == "out_of_domain":
                return self._create_early_result(ctx)
            if ctx.is_fallback and ctx.fallback_reason == "nan_or_inf_rejected":
                return self._create_sanitize_fallback_result(ctx)
            if ctx.is_fallback and ctx.diagnostic is not None:
                return self._create_fallback_result(ctx)
            if _total_elapsed_ms > timer.budget_ms:
                logger.warning(
                    "pipeline_over_budget",
                    extra={
                        "series_id": series_id,
                        "phase": phase_name,
                        "elapsed_ms": round(_total_elapsed_ms, 2),
                        "budget_ms": timer.budget_ms,
                    },
                )
                ctx.metadata["_cognitive_phase_times"] = _phase_times
                ctx.metadata["truncated"] = True
                ctx.metadata["truncation_reason"] = "over_budget"
                return self._create_over_budget_result(ctx)
        ctx.metadata["_cognitive_phase_times"] = _phase_times
        return self._assembly.execute(ctx)
    def _create_early_result(self, ctx: PipelineContext) -> PredictionResult:
        return BoundaryCheckPhase().create_early_result(ctx)
    def _create_fallback_result(self, ctx: PipelineContext) -> PredictionResult:
        from ...interfaces import PredictionResult
        return PredictionResult(
            predicted_value=ctx.orchestrator._last_explanation.predicted_value if ctx.orchestrator._last_explanation else None,
            confidence=0.2, trend="unknown", metadata=ctx.metadata)
    def _create_over_budget_result(self, ctx: PipelineContext) -> PredictionResult:
        from ...interfaces import PredictionResult
        return PredictionResult(
            predicted_value=ctx.orchestrator._last_explanation.predicted_value if ctx.orchestrator._last_explanation else None,
            confidence=0.15,
            trend="unknown",
            metadata={
                **ctx.metadata,
                "is_over_budget_fallback": True,
                "rejection_reason": "over_budget",
            },
        )
    def _create_sanitize_fallback_result(self, ctx: PipelineContext) -> PredictionResult:
        from ...interfaces import PredictionResult
        return PredictionResult(
            predicted_value=None, confidence=0.0, trend="unknown",
            metadata={"is_sanitize_fallback": True, "rejection_reason": "nan_or_inf_rejected",
                      "sanitization_flags": list(ctx.sanitization_flags)})

def execute_pipeline(orchestrator, values: List[float], timestamps: Optional[List[float]],
                     series_id: str, flags_snapshot: Optional[Any]=None) -> PredictionResult:
    """Entry point para ejecutar el pipeline."""
    from .pipeline_executor_factory import PipelineExecutorFactory
    factory = getattr(orchestrator, "_pipeline_executor_factory", None)
    if not isinstance(factory, PipelineExecutorFactory):
        factory = PipelineExecutorFactory()
    executor = factory.create(flags_snapshot)
    return executor.execute(orchestrator=orchestrator, values=values, timestamps=timestamps,
                            series_id=series_id, flags=flags_snapshot)
