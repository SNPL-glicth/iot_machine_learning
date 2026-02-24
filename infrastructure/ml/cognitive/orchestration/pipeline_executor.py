from __future__ import annotations

import concurrent.futures
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional
    from ...interfaces import PredictionResult
    from ..types import PipelineTimer

from ..explanation_builder import ExplanationBuilder
from ..orchestrator_helpers import collect_perceptions
from ..analysis.types import MetaDiagnostic, PipelineTimer
from .fallback_handler import handle_fallback

logger = logging.getLogger(__name__)


def execute_pipeline(
    orchestrator,
    values: List[float],
    timestamps: Optional[List[float]],
    series_id: str,
) -> PredictionResult:
    """Execute full cognitive pipeline.
    
    Args:
        orchestrator: MetaCognitiveOrchestrator instance
        values: Time series values
        timestamps: Optional timestamps
        series_id: Series identifier
    
    Returns:
        PredictionResult with cognitive metadata
    """
    timer = PipelineTimer(budget_ms=orchestrator._budget_ms)

    # Phase: PERCEIVE
    timer.start()
    profile = orchestrator._analyzer.analyze(values, timestamps)
    regime_str = profile.regime.value
    builder = ExplanationBuilder(series_id)
    builder.set_signal(profile)
    
    neighbor_trends = {}
    if orchestrator._correlation_port and series_id != "unknown":
        try:
            neighbors = orchestrator._correlation_port.get_correlated_series(series_id, max_neighbors=3)
            if neighbors:
                neighbor_ids = [n[0] for n in neighbors]
                neighbor_values = orchestrator._correlation_port.get_recent_values_multi(neighbor_ids, window=5)
                for nid, nvals in neighbor_values.items():
                    if len(nvals) >= 2:
                        slope = (nvals[-1] - nvals[0]) / max(len(nvals) - 1, 1)
                        neighbor_trends[nid] = "up" if slope > 0.1 else "down" if slope < -0.1 else "stable"
        except Exception as e:
            logger.debug(f"correlation enrichment failed: {e}")
    
    timer.stop("perceive")
    
    if orchestrator._enable_advanced_plasticity and orchestrator._plasticity_coordinator:
        orchestrator._last_plasticity_context = orchestrator._plasticity_coordinator.create_plasticity_context(
            profile, series_id
        )
    else:
        orchestrator._last_plasticity_context = None

    # Phase: PREDICT
    timer.start()
    perceptions = collect_perceptions(orchestrator._engines, values, timestamps)
    timer.stop("predict")

    if not perceptions:
        orchestrator._last_timer = timer
        result, diag, expl, reg, perc = handle_fallback(values, profile, builder, timer, "no_valid_perceptions")
        orchestrator._last_diagnostic = diag
        orchestrator._last_explanation = expl
        orchestrator._last_regime = reg
        orchestrator._last_perceptions = perc
        return result

    if timer.total_ms > timer.budget_ms:
        logger.warning("pipeline_budget_exceeded", extra={
            "phase": "predict", "elapsed_ms": round(timer.total_ms, 2),
            "budget_ms": timer.budget_ms,
        })
        orchestrator._last_timer = timer
        result, diag, expl, reg, perc = handle_fallback(values, profile, builder, timer, "budget_exceeded")
        orchestrator._last_diagnostic = diag
        orchestrator._last_explanation = expl
        orchestrator._last_regime = reg
        orchestrator._last_perceptions = perc
        return result

    builder.set_perceptions(perceptions, n_engines_total=len(orchestrator._engines))

    # Phase: ADAPT
    timer.start()
    error_dict = {k: list(v) for k, v in orchestrator._recent_errors.items()}
    engine_names = [p.engine_name for p in perceptions]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            adapt_future = executor.submit(
                orchestrator._weight_service.resolve_weights,
                regime_str, engine_names, series_id
            )
            plasticity_weights = adapt_future.result(timeout=0.1)
    except (concurrent.futures.TimeoutError, Exception) as e:
        logger.debug(f"parallel adapt failed, falling back to sequential: {e}")
        plasticity_weights = orchestrator._weight_service.resolve_weights(regime_str, engine_names, series_id)

    adapted = (orchestrator._plasticity is not None and orchestrator._plasticity.has_history(regime_str))
    builder.set_adaptation(adapted=adapted, regime=regime_str)
    timer.stop("adapt")

    # Phase: INHIBIT
    timer.start()
    inh_states = orchestrator._inhibition.compute(perceptions, plasticity_weights, error_dict)
    timer.stop("inhibit")
    
    mediated_weights = orchestrator._weight_mediator.mediate(plasticity_weights, inh_states)
    builder.set_inhibition(inh_states, mediated_weights)

    # Phase: FUSE
    timer.start()
    (fused_val, fused_conf, fused_trend,
     final_weights, selected, reason) = orchestrator._fusion.fuse(
        perceptions, inh_states, neighbor_trends=neighbor_trends, signal_std=profile.std)

    method = "weighted_average" if len(perceptions) > 1 else "single_engine"
    builder.set_fusion(
        fused_val, fused_conf, fused_trend,
        final_weights, selected, reason, method,
    )
    timer.stop("fuse")

    # Phase: EXPLAIN
    timer.start()
    diag = MetaDiagnostic(
        signal_profile=profile, perceptions=perceptions,
        inhibition_states=inh_states, final_weights=final_weights,
        selected_engine=selected, selection_reason=reason,
        fusion_method=method,
    )
    orchestrator._last_diagnostic = diag
    orchestrator._last_explanation = builder.build()
    timer.stop("explain")

    orchestrator._last_regime = regime_str
    orchestrator._last_perceptions = list(perceptions)
    orchestrator._last_timer = timer

    logger.debug("cognitive_prediction", extra={
        "n_engines": len(perceptions), "selected": selected,
        "regime": regime_str, "fused_value": round(fused_val, 4),
        "pipeline_ms": round(timer.total_ms, 2),
    })
    if timer.is_over_budget:
        logger.warning("pipeline_over_budget", extra=timer.to_dict())

    metadata = {
        "cognitive_diagnostic": diag.to_dict(),
        "explanation": orchestrator._last_explanation.to_dict(),
        "pipeline_timing": timer.to_dict(),
    }
    
    if orchestrator._storage and series_id != "unknown":
        ci = orchestrator._storage.compute_confidence_interval(
            series_id, selected, fused_val
        )
        if ci:
            metadata["confidence_interval"] = ci

    from ...interfaces import PredictionResult
    return PredictionResult(
        predicted_value=fused_val, confidence=fused_conf,
        trend=fused_trend,
        metadata=metadata,
    )
