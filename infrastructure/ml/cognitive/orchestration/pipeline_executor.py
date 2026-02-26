from __future__ import annotations

import concurrent.futures
import logging
import time
from collections import OrderedDict
from threading import RLock
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from typing import List, Optional
    from ...interfaces import PredictionResult
    from ..types import PipelineTimer

from ..explanation_builder import ExplanationBuilder
from ..orchestrator_helpers import collect_perceptions
from ..analysis.types import MetaDiagnostic, PipelineTimer
from .fallback_handler import handle_fallback

logger = logging.getLogger(__name__)


class WeightCache:
    def __init__(self, max_entries: int = 1000, ttl_seconds: float = 60.0):
        self._cache: OrderedDict[Tuple[str, str], Tuple[Dict[str, float], float]] = OrderedDict()
        self._lock = RLock()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
    
    def get(self, series_id: str, regime: str) -> Optional[Dict[str, float]]:
        with self._lock:
            key = (series_id, regime)
            if key not in self._cache:
                return None
            weights, timestamp = self._cache[key]
            if time.time() - timestamp >= self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return weights
    
    def set(self, series_id: str, regime: str, weights: Dict[str, float]):
        with self._lock:
            key = (series_id, regime)
            while len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = (weights, time.time())


_weight_cache = WeightCache()


def _compute_robust_gradient(values: list) -> float:
    """Compute OLS gradient over values."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if abs(den) > 1e-9 else 0.0


def _apply_spatial_correction(
    base_prediction: float,
    neighbors: list,
    neighbor_values: dict,
    max_correction_pct: float = 0.15,
    min_gradient_samples: int = 3,
) -> float:
    """Apply spatial gradient correction from correlated neighbors.
    
    Computes weighted gradient: Σ(correlation_i × gradient_i) / Σ|correlation_i|
    Correction: gradient × (Σ|correlation_i| / n_neighbors)
    
    Args:
        base_prediction: Base prediction value
        neighbors: List of (neighbor_id, correlation) tuples
        neighbor_values: Dict of {neighbor_id: [values]}
        max_correction_pct: Maximum correction as percentage of base (default 15%)
        min_gradient_samples: Minimum samples required for gradient (default 3)
    
    Returns:
        Corrected prediction value
    """
    if not neighbors:
        return base_prediction
    
    valid_neighbors = []
    for neighbor_id, correlation in neighbors:
        if abs(correlation) > 0.5 and neighbor_id in neighbor_values:
            values = neighbor_values[neighbor_id]
            if len(values) >= min_gradient_samples:
                gradient = _compute_robust_gradient(values)
                valid_neighbors.append((neighbor_id, correlation, gradient))
    
    if not valid_neighbors:
        return base_prediction
    
    weighted_gradient = 0.0
    total_abs_correlation = 0.0
    
    for neighbor_id, correlation, gradient in valid_neighbors:
        weighted_gradient += correlation * gradient
        total_abs_correlation += abs(correlation)
    
    if total_abs_correlation < 1e-9:
        return base_prediction
    
    gradient = weighted_gradient / total_abs_correlation
    correction = gradient * (total_abs_correlation / len(valid_neighbors))
    
    max_correction = abs(base_prediction) * max_correction_pct
    correction = max(-max_correction, min(max_correction, correction))
    
    return base_prediction + correction


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
    neighbors = []
    neighbor_values_dict = {}
    if orchestrator._correlation_port and series_id != "unknown":
        try:
            neighbors = orchestrator._correlation_port.get_correlated_series(series_id, max_neighbors=3)
            if neighbors:
                neighbor_ids = [n[0] for n in neighbors]
                neighbor_values_dict = orchestrator._correlation_port.get_recent_values_multi(neighbor_ids, window=5)
                for nid, nvals in neighbor_values_dict.items():
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

    plasticity_weights = _weight_cache.get(series_id, regime_str)
    if plasticity_weights is None:
        plasticity_weights = orchestrator._weight_service.resolve_weights(regime_str, engine_names, series_id)
        _weight_cache.set(series_id, regime_str, plasticity_weights)

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

    fused_val = _apply_spatial_correction(fused_val, neighbors, neighbor_values_dict)
    
    if orchestrator._correlation_port and neighbors:
        high_corr_neighbors = [(nid, corr) for nid, corr in neighbors if abs(corr) > 0.7]
        if high_corr_neighbors:
            try:
                predictions_to_smooth = {series_id: fused_val}
                for neighbor_id, _ in high_corr_neighbors:
                    if orchestrator._storage:
                        neighbor_pred = orchestrator._storage.get_latest_prediction_for_series(neighbor_id)
                        if neighbor_pred and hasattr(neighbor_pred, 'predicted_value'):
                            predictions_to_smooth[neighbor_id] = neighbor_pred.predicted_value
                
                if len(predictions_to_smooth) >= 2:
                    smoothed = orchestrator._correlation_port.smooth_with_field(
                        predictions_to_smooth,
                        smoothing_factor=0.2,
                    )
                    if smoothed and series_id in smoothed:
                        original_val = fused_val
                        fused_val = smoothed[series_id]
                        logger.debug("field_smoothing_applied", extra={
                            "original": round(original_val, 4),
                            "smoothed": round(fused_val, 4),
                            "n_neighbors": len(predictions_to_smooth) - 1,
                        })
            except Exception as e:
                logger.debug(f"field_smoothing_failed: {e}")

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
