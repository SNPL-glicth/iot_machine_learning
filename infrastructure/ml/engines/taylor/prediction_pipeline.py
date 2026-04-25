"""Taylor prediction pipeline — extracted from engine.py."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from iot_machine_learning.domain.validators.numeric import (
    clamp_prediction,
    validate_window,
)
from iot_machine_learning.domain.entities.structural_analysis import StructuralAnalysis
from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult

from .types import DerivativeMethod
from .diagnostics import compute_diagnostic
from .time_step import compute_dt
from .polynomial import project
from .coefficient_cache import TaylorCoefficientCache
from .engine_helpers import (
    sanitize_inputs,
    load_hyperparams,
    compute_taylor_coefficients,
    classify_trend,
    confidence_from_stability,
    taylor_fallback,
)

if TYPE_CHECKING:
    from .engine import TaylorPredictionEngine

logger = logging.getLogger(__name__)


def run_taylor_prediction(
    engine: "TaylorPredictionEngine",
    values: List[float],
    timestamps: Optional[List[float]] = None,
) -> PredictionResult:
    """Execute the full Taylor prediction pipeline."""
    values, timestamps = sanitize_inputs(values, timestamps)
    if not values:
        return PredictionResult(
            predicted_value=None,
            confidence=0.0,
            trend="unknown",
            metadata={"reason": "all_inputs_invalid"},
        )

    load_hyperparams(engine, window_size=len(values))
    validate_window(values, min_size=1)
    n = len(values)

    # FASE 2: Gap detection — use largest continuous segment
    if engine._gap_detector and timestamps and len(timestamps) == len(values):
        values, timestamps = engine._gap_detector.get_largest_segment(
            values, timestamps
        )
        if len(values) < n:
            logger.info(
                "taylor_gap_segmentation",
                extra={"original_size": n, "segment_size": len(values)},
            )
            n = len(values)

    # FASE 2: Robust Δt computation (gap-aware)
    if engine._gap_detector and timestamps:
        dt = engine._gap_detector.compute_robust_dt(timestamps)
        if dt is None:
            dt = compute_dt(timestamps)
    else:
        dt = compute_dt(timestamps)

    if not engine.can_handle(n):
        return taylor_fallback(values, engine._min_confidence, engine._horizon)

    # FASE 2: Check cache first
    window_hash = None
    if engine._cache and engine._series_id:
        window_hash = TaylorCoefficientCache.compute_window_hash(values, timestamps)
        cached = engine._cache.get(engine._series_id, window_hash)
        if cached:
            coeffs = cached
            logger.debug("taylor_cache_hit", extra={"series_id": engine._series_id})
        else:
            coeffs = compute_taylor_coefficients(
                values, dt, engine._order, engine._method
            )
            engine._cache.put(engine._series_id, coeffs, window_hash, n, dt)
    else:
        coeffs = compute_taylor_coefficients(
            values, dt, engine._order, engine._method
        )

    if coeffs.estimated_order == 0:
        return taylor_fallback(values, engine._min_confidence, engine._horizon)

    h = float(engine._horizon) * dt
    predicted_raw = project(coeffs, h, coeffs.estimated_order)
    predicted, clamped = clamp_prediction(
        predicted_raw, values, margin_pct=engine._clamp_margin_pct,
    )

    # FIX-9: Apply physical bounds if configured
    physical_clamp_applied = False
    physical_clamp_direction = None
    if engine._physical_min is not None and predicted < engine._physical_min:
        predicted = engine._physical_min
        physical_clamp_applied = True
        physical_clamp_direction = "min"
    if engine._physical_max is not None and predicted > engine._physical_max:
        predicted = engine._physical_max
        physical_clamp_applied = True
        physical_clamp_direction = "max"

    trend = classify_trend(coeffs.local_slope, engine._trend_threshold)
    diag = compute_diagnostic(coeffs, values, dt)
    base_confidence = confidence_from_stability(
        diag.stability_indicator,
        engine._min_confidence,
        engine._max_confidence,
    )

    # FASE 2: Adjust confidence with historical error
    if engine._tracker:
        value_range = max(values) - min(values) if len(values) > 1 else 1.0
        confidence = engine._tracker.compute_confidence_adjustment(
            base_confidence, value_range
        )
    else:
        confidence = base_confidence

    structural = StructuralAnalysis.from_taylor_diagnostic(diag, values)

    metadata: Dict = {
        "order": coeffs.estimated_order,
        "derivatives": coeffs.to_dict(),
        "dt": dt,
        "horizon_steps": engine._horizon,
        "fallback": None,
        "clamped": clamped,
        "physical_clamp_applied": physical_clamp_applied,
        "physical_clamp_direction": physical_clamp_direction,
        "diagnostic": diag.to_dict(),
        "structural_analysis": structural.to_dict(),
        "cache_hit": window_hash is not None and engine._cache is not None,
    }

    # FASE 2: Add performance metrics to metadata
    if engine._tracker:
        perf_metrics = engine._tracker.get_metrics()
        if perf_metrics:
            metadata["performance"] = {
                "mae": round(perf_metrics.mae, 4),
                "rmse": round(perf_metrics.rmse, 4),
                "recent_mae": round(perf_metrics.recent_mae, 4),
                "n_samples": perf_metrics.n_samples,
            }
            metadata["confidence_base"] = round(base_confidence, 4)
            metadata["confidence_adjusted"] = round(confidence, 4)

    logger.debug(
        "taylor_prediction",
        extra={
            "n_points": n,
            "effective_order": coeffs.estimated_order,
            "method": coeffs.method,
            "predicted": predicted,
            "clamped": clamped,
            "confidence": confidence,
            "stability": diag.stability_indicator,
        },
    )

    return PredictionResult(
        predicted_value=predicted,
        confidence=confidence,
        trend=trend,
        metadata=metadata,
    )
