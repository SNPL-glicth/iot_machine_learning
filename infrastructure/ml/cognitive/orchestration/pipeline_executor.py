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

from ..explanation import ExplanationBuilder
from ..perception.helpers import collect_perceptions
from ..analysis.types import MetaDiagnostic, PipelineTimer
from .fallback_handler import handle_fallback
from .....domain.services.signal_coherence_checker import (
    SignalCoherenceChecker,
    CoherenceResult,
)
from .....domain.services.engine_decision_arbiter import (
    EngineDecisionArbiter,
    EngineDecision,
)
from .....domain.services.confidence_calibrator import (
    ConfidenceCalibrator,
    CalibratedConfidence,
)
from .....domain.services.action_guard import (
    ActionGuard,
    GuardedAction,
)
from .....domain.services.domain_boundary_checker import (
    DomainBoundaryChecker,
)
from .....domain.entities.results.boundary_result import (
    BoundaryResult,
)
from .....domain.entities.results.unified_narrative import (
    UnifiedNarrative,
)
from .....domain.services.narrative_unifier import (
    NarrativeUnifier,
)

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
    flags_snapshot: Optional[Any] = None,
) -> PredictionResult:
    """Execute full cognitive pipeline.
    
    Args:
        orchestrator: MetaCognitiveOrchestrator instance
        values: Time series values
        timestamps: Optional timestamps
        series_id: Series identifier
        flags_snapshot: Optional pre-captured feature flags snapshot.
            If None, flags are captured at pipeline start (consistent throughout).
    
    Returns:
        PredictionResult with cognitive metadata
    """
    # Capture flags once at pipeline start for consistency
    if flags_snapshot is None:
        from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags
        flags = get_feature_flags()
    else:
        flags = flags_snapshot
    
    timer = PipelineTimer(budget_ms=orchestrator._budget_ms)

    # Phase: DOMAIN BOUNDARY CHECK (EJE 4 fix — only if enabled, first phase)
    boundary_result: Optional[BoundaryResult] = None
    try:
        if flags.ML_DOMAIN_BOUNDARY_ENABLED:
            checker = DomainBoundaryChecker()
            # Get noise ratio from profile if available (placeholder for now)
            noise_ratio = 0.0
            boundary_result = checker.check(
                values=values,
                timestamps=timestamps,
                noise_ratio=noise_ratio,
            )
            
            if not boundary_result.within_domain:
                logger.warning("domain_boundary_violation", extra={
                    "series_id": series_id,
                    "rejection_reason": boundary_result.rejection_reason,
                    "n_points": len(values),
                })
                # Return immediate out-of-domain result
                from ...interfaces import PredictionResult
                return PredictionResult(
                    predicted_value=None,
                    confidence=0.0,
                    trend="unknown",
                    metadata={
                        "is_out_of_domain": True,
                        "rejection_reason": boundary_result.rejection_reason,
                        "boundary_check": {
                            "within_domain": False,
                            "rejection_reason": boundary_result.rejection_reason,
                            "data_quality_score": 0.0,
                            "warnings": [],
                        },
                    },
                )
    except Exception as e:
        logger.debug(f"domain_boundary_check_skipped: {e}")

    # Phase: PERCEIVE
    timer.start()
    profile = orchestrator._analyzer.analyze(values, timestamps)
    regime_str = profile.regime.value if hasattr(profile.regime, 'value') else str(profile.regime)
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
        # Phase 3: Use WeightResolutionService (consolidated weight logic)
        plasticity_weights = orchestrator._weight_resolver.resolve(
            regime=regime_str,
            engine_names=engine_names,
            series_id=series_id,
        )
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

    # Phase: DECISION ARBITER (EJE 1 fix — only if enabled)
    engine_decision: Optional[EngineDecision] = None
    try:
        if flags.ML_DECISION_ARBITER_ENABLED:
            # Determine engines from each layer
            flag_engine = flags.get_active_engine_for_series(series_id) if hasattr(flags, 'get_active_engine_for_series') else "unknown"
            profile_engine = selected  # From fusion/WeightedFusion
            fusion_engine = selected   # From fusion
            
            arbiter = EngineDecisionArbiter()
            engine_decision = arbiter.arbitrate(
                flag_engine=flag_engine,
                profile_engine=profile_engine,
                fusion_engine=fusion_engine,
                series_id=series_id,
                rollback_to_baseline=flags.ML_ROLLBACK_TO_BASELINE,
                series_overrides=getattr(flags, 'ML_ENGINE_SERIES_OVERRIDES', {}),
            )
            logger.info("engine_decision_arbitrated", extra={
                "series_id": series_id,
                "chosen_engine": engine_decision.chosen_engine,
                "authority": engine_decision.authority,
                "reason": engine_decision.reason,
                "overrides": engine_decision.overrides,
            })
    except Exception as e:
        logger.debug(f"engine_decision_arbiter_skipped: {e}")

    # Phase: COHERENCE CHECK (EJE 2 fix — only if enabled)
    coherence_result: Optional[CoherenceResult] = None
    try:
        if flags.ML_COHERENCE_CHECK_ENABLED:
            # Note: In real implementation, anomaly_result would come from
            # anomaly detection service. Here we use placeholder for structure.
            checker = SignalCoherenceChecker()
            # Use profile values as proxy for historical range
            historical = values if len(values) > 0 else None
            # Placeholder: assume no anomaly for now (anomaly integration separate)
            is_anomaly = False
            anomaly_score = 0.0
            coherence_result = checker.check(
                predicted_value=fused_val,
                predicted_confidence=fused_conf,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                historical_values=historical,
            )
            if not coherence_result.is_coherent:
                fused_conf = coherence_result.resolved_confidence
                logger.warning("coherence_conflict_detected", extra={
                    "conflict_type": coherence_result.conflict_type,
                    "resolved_confidence": round(fused_conf, 4),
                    "reason": coherence_result.resolution_reason,
                })
    except Exception as e:
        logger.debug(f"coherence_check_skipped: {e}")

    # Phase: CONFIDENCE CALIBRATION (EJE 6 fix — only if enabled)
    calibrated_confidence: Optional[CalibratedConfidence] = None
    try:
        if flags.ML_CONFIDENCE_CALIBRATION_ENABLED:
            calibrator = ConfidenceCalibrator()
            
            # Compute engine disagreement from perceptions
            engine_disagreement = calibrator.compute_engine_disagreement(perceptions)
            
            # Determine if only baseline is active
            only_baseline = len(perceptions) == 1 and perceptions[0].engine_name == "baseline"
            all_inhibited = all(s.inhibited_weight < 0.05 for s in inh_states) if inh_states else False
            
            # Get noise ratio from profile if available
            noise_ratio = getattr(profile, 'noise_ratio', 0.0)
            
            # Check coherence conflict from EJE 2
            coherence_conflict = coherence_result is not None and not coherence_result.is_coherent
            
            calibrated_confidence = calibrator.calibrate(
                raw_confidence=fused_conf,
                n_points=len(values),
                noise_ratio=noise_ratio,
                engine_disagreement=engine_disagreement,
                only_baseline_active=only_baseline,
                coherence_conflict=coherence_conflict,
                all_engines_inhibited=all_inhibited,
            )
            
            # Replace fused_conf with calibrated
            fused_conf = calibrated_confidence.calibrated
            
            logger.info("confidence_calibrated", extra={
                "series_id": series_id,
                "raw": round(calibrated_confidence.raw, 4),
                "calibrated": round(calibrated_confidence.calibrated, 4),
                "penalty": round(calibrated_confidence.penalty_applied, 4),
                "n_reasons": len(calibrated_confidence.reasons),
            })
    except Exception as e:
        logger.debug(f"confidence_calibration_skipped: {e}")

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

    # Phase: ACTION GUARD (EJE 5 fix — only if enabled, last phase)
    guarded_action: Optional[GuardedAction] = None
    try:
        if flags.ML_ACTION_GUARD_ENABLED:
            # Determine series state (placeholder: use "UNKNOWN" if not available)
            series_state = getattr(profile, 'series_state', 'UNKNOWN')
            
            # Extract action info from explanation if available
            action_required = False
            recommended_action = None
            severity = "NORMAL"
            
            if orchestrator._last_explanation:
                # Try to extract severity and action from explanation
                outcome = getattr(orchestrator._last_explanation, 'outcome', None)
                if outcome:
                    severity = getattr(outcome, 'severity', 'NORMAL')
                    action_required = getattr(outcome, 'action_required', False)
                    recommended_action = getattr(outcome, 'recommended_action', None)
            
            guard = ActionGuard()
            guarded_action = guard.guard(
                action_required=action_required,
                recommended_action=recommended_action,
                severity=severity,
                series_state=series_state,
            )
            
            if not guarded_action.action_allowed:
                logger.warning("action_suppressed", extra={
                    "series_id": series_id,
                    "series_state": series_state,
                    "original_action": recommended_action,
                    "reason": guarded_action.suppressed_reason,
                })
    except Exception as e:
        logger.debug(f"action_guard_skipped: {e}")

    metadata = {
        "cognitive_diagnostic": diag.to_dict(),
        "explanation": orchestrator._last_explanation.to_dict(),
        "pipeline_timing": timer.to_dict(),
    }
    
    # Add boundary check result if available (with warnings)
    if boundary_result is not None and boundary_result.within_domain:
        metadata["boundary_check"] = {
            "within_domain": True,
            "data_quality_score": boundary_result.data_quality_score,
            "warnings": boundary_result.warnings,
        }
    
    # Add coherence check result if available
    if coherence_result is not None:
        metadata["coherence_check"] = {
            "is_coherent": coherence_result.is_coherent,
            "conflict_type": coherence_result.conflict_type,
            "resolved_confidence": coherence_result.resolved_confidence,
            "resolution_reason": coherence_result.resolution_reason,
        }
    
    # Add engine decision if available
    if engine_decision is not None:
        metadata["engine_decision"] = {
            "chosen_engine": engine_decision.chosen_engine,
            "authority": engine_decision.authority,
            "reason": engine_decision.reason,
            "overrides": engine_decision.overrides,
        }
    
    # Add confidence calibration report if available
    if calibrated_confidence is not None:
        metadata["calibration_report"] = {
            "raw": calibrated_confidence.raw,
            "calibrated": calibrated_confidence.calibrated,
            "penalty_applied": calibrated_confidence.penalty_applied,
            "reasons": calibrated_confidence.reasons,
        }
    
    # Add action guard result if available
    if guarded_action is not None:
        metadata["action_guard"] = {
            "action_allowed": guarded_action.action_allowed,
            "original_action": guarded_action.original_action,
            "final_action": guarded_action.final_action,
            "suppressed_reason": guarded_action.suppressed_reason,
            "series_state": guarded_action.series_state,
        }
    
    # Phase: NARRATIVE UNIFICATION (EJE 7 fix — only if enabled, absolute last phase)
    unified_narrative: Optional[UnifiedNarrative] = None
    try:
        if flags.ML_NARRATIVE_UNIFICATION_ENABLED:
            unifier = NarrativeUnifier()
            
            # Build narrative sources from available data
            prediction_explanation = None
            anomaly_narrative = None
            text_narrative = None
            
            # Extract from explanation builder result
            if orchestrator._last_explanation:
                outcome = getattr(orchestrator._last_explanation, 'outcome', None)
                if outcome:
                    prediction_explanation = {
                        "verdict": getattr(outcome, 'description', None),
                        "severity": getattr(outcome, 'severity', 'UNKNOWN'),
                        "confidence": fused_conf,
                    }
            
            # Anomaly narrative placeholder (would come from anomaly detection)
            # For now, use coherence result as proxy if conflict detected
            if coherence_result and not coherence_result.is_coherent:
                anomaly_narrative = {
                    "verdict": f"coherence_conflict:{coherence_result.conflict_type}",
                    "severity": "WARNING",
                    "confidence": coherence_result.resolved_confidence,
                }
            
            unified_narrative = unifier.unify(
                prediction_explanation=prediction_explanation,
                anomaly_narrative=anomaly_narrative,
                text_narrative=text_narrative,
            )
            
            # Log if contradictions detected
            if unified_narrative.contradictions:
                logger.warning("narrative_contradictions_detected", extra={
                    "series_id": series_id,
                    "contradictions": unified_narrative.contradictions,
                    "unified_severity": unified_narrative.severity,
                })
    except Exception as e:
        logger.debug(f"narrative_unification_skipped: {e}")

    # Add unified narrative to metadata if available
    if unified_narrative is not None:
        metadata["unified_narrative"] = {
            "primary_verdict": unified_narrative.primary_verdict,
            "severity": unified_narrative.severity,
            "confidence": unified_narrative.confidence,
            "contradictions": unified_narrative.contradictions,
            "sources_used": unified_narrative.sources_used,
            "suppressed": unified_narrative.suppressed,
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
