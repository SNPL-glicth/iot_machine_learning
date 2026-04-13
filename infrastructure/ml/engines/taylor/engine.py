"""Taylor prediction engine — orchestrator.

Delegates all math to the ``taylor/`` package and focuses on:
1. Input validation
2. Order negotiation
3. Clamping raw predictions to observed range
4. Trend classification from first derivative
5. Confidence estimation from stability indicator
6. Assembling metadata with TaylorDiagnostic

FASE 2 enhancements:
- Coefficient caching with TTL (fixes CRIT-1)
- MAE/RMSE tracking for confidence adjustment (fixes CRIT-2)
- Temporal gap detection (fixes CRIT-3)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from iot_machine_learning.domain.validators.numeric import (
    clamp_prediction,
    validate_window,
)
from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)

from .types import DerivativeMethod
from .diagnostics import compute_diagnostic
from .time_step import compute_dt
from .derivatives import estimate_derivatives
from .polynomial import project
from .coefficient_cache import TaylorCoefficientCache
from .performance_tracker import TaylorPerformanceTracker
from .gap_detector import TemporalGapDetector

logger = logging.getLogger(__name__)

_TREND_THRESHOLD: float = 0.01
_MIN_CONFIDENCE: float = 0.3
_MAX_CONFIDENCE: float = 0.95
_CLAMP_MARGIN_PCT: float = 0.3
_VARIANCE_EPSILON: float = 1e-9  # CRIT-4: Variance threshold for order=0


class TaylorPredictionEngine(PredictionEngine):
    """Taylor-series prediction engine.

    See ``taylor/`` package for mathematical specification.
    Configurable derivative method: backward, central, least_squares.
    
    FASE 2 features:
    - Coefficient caching (enable_cache=True)
    - Performance tracking for confidence (enable_tracking=True)
    - Gap detection (enable_gap_detection=True)
    """

    def __init__(
        self,
        order: int = 2,
        horizon: int = 1,
        *,
        derivative_method: DerivativeMethod = DerivativeMethod.BACKWARD,
        trend_threshold: float = _TREND_THRESHOLD,
        min_confidence: float = _MIN_CONFIDENCE,
        max_confidence: float = _MAX_CONFIDENCE,
        clamp_margin_pct: float = _CLAMP_MARGIN_PCT,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 300,
        enable_tracking: bool = True,
        enable_gap_detection: bool = True,
        series_id: Optional[str] = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon debe ser >= 1, recibido {horizon}")
        self._order = max(1, min(order, 3))
        self._horizon = horizon
        self._method = derivative_method
        self._trend_threshold = trend_threshold
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence
        self._clamp_margin_pct = clamp_margin_pct
        self._series_id = series_id
        
        # FASE 2: Coefficient cache
        self._cache = TaylorCoefficientCache(ttl_seconds=cache_ttl_seconds) if enable_cache else None
        
        # FASE 2: Performance tracker
        self._tracker = TaylorPerformanceTracker() if enable_tracking else None
        
        # FASE 2: Gap detector
        self._gap_detector = TemporalGapDetector() if enable_gap_detection else None
        
        if order != self._order:
            logger.warning(
                "taylor_order_clamped",
                extra={"requested": order, "effective": self._order},
            )

    @property
    def name(self) -> str:
        return "taylor_finite_differences"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= self._order + 2

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        validate_window(values, min_size=1)
        n = len(values)
        
        # FASE 2: Gap detection — use largest continuous segment
        if self._gap_detector and timestamps and len(timestamps) == len(values):
            values, timestamps = self._gap_detector.get_largest_segment(values, timestamps)
            if len(values) < n:
                logger.info(
                    "taylor_gap_segmentation",
                    extra={
                        "original_size": n,
                        "segment_size": len(values),
                    },
                )
                n = len(values)
        
        # FASE 2: Robust Δt computation (gap-aware)
        if self._gap_detector and timestamps:
            dt = self._gap_detector.compute_robust_dt(timestamps)
            if dt is None:
                dt = compute_dt(timestamps)
        else:
            dt = compute_dt(timestamps)

        if not self.can_handle(n):
            return self._fallback(values)

        # FASE 2: Check cache first
        window_hash = None
        if self._cache and self._series_id:
            window_hash = TaylorCoefficientCache.compute_window_hash(values, timestamps)
            cached_coeffs = self._cache.get(self._series_id, window_hash)
            if cached_coeffs:
                coeffs = cached_coeffs
                logger.debug("taylor_cache_hit", extra={"series_id": self._series_id})
            else:
                coeffs = self._compute_coefficients(values, dt)
                self._cache.put(self._series_id, coeffs, window_hash, n, dt)
        else:
            coeffs = self._compute_coefficients(values, dt)
        if coeffs.estimated_order == 0:
            return self._fallback(values)

        h = float(self._horizon) * dt
        predicted_raw = project(coeffs, h, coeffs.estimated_order)
        predicted, clamped = clamp_prediction(
            predicted_raw, values, margin_pct=self._clamp_margin_pct,
        )
        trend = self._classify_trend(coeffs.local_slope)
        diag = compute_diagnostic(coeffs, values, dt)
        base_confidence = self._confidence_from_stability(diag.stability_indicator)
        
        # FASE 2: Adjust confidence with historical error
        if self._tracker:
            value_range = max(values) - min(values) if len(values) > 1 else 1.0
            confidence = self._tracker.compute_confidence_adjustment(
                base_confidence, value_range
            )
        else:
            confidence = base_confidence

        from iot_machine_learning.domain.entities.structural_analysis import (
            StructuralAnalysis,
        )

        structural = StructuralAnalysis.from_taylor_diagnostic(diag, values)

        metadata: dict = {
            "order": coeffs.estimated_order,
            "derivatives": coeffs.to_dict(),
            "dt": dt,
            "horizon_steps": self._horizon,
            "fallback": None,
            "clamped": clamped,
            "diagnostic": diag.to_dict(),
            "structural_analysis": structural.to_dict(),
            "cache_hit": window_hash is not None and self._cache is not None,
        }
        
        # FASE 2: Add performance metrics to metadata
        if self._tracker:
            perf_metrics = self._tracker.get_metrics()
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
                "n_points": n, "effective_order": coeffs.estimated_order,
                "method": coeffs.method, "predicted": predicted,
                "clamped": clamped, "confidence": confidence,
                "stability": diag.stability_indicator,
            },
        )
        return PredictionResult(
            predicted_value=predicted, confidence=confidence,
            trend=trend, metadata=metadata,
        )

    def supports_uncertainty(self) -> bool:
        return False
    
    def record_actual(self, predicted: float, actual: float) -> None:
        """Record actual value for performance tracking (FASE 2).
        
        Args:
            predicted: Predicted value
            actual: Actual observed value
        """
        if self._tracker:
            self._tracker.record_error(predicted, actual)
            logger.debug(
                "taylor_actual_recorded",
                extra={
                    "predicted": round(predicted, 4),
                    "actual": round(actual, 4),
                    "error": round(abs(predicted - actual), 4),
                },
            )

    # -- private helpers --------------------------------------------------
    
    def _compute_coefficients(self, values: List[float], dt: float):
        """Compute Taylor coefficients with variance check (CRIT-4)."""
        variance = self._compute_variance(values)
        effective_order = self._order
        
        if variance < _VARIANCE_EPSILON:
            # Signal is essentially constant, force order=0
            effective_order = 0
            logger.debug(
                "taylor_order_reduced_to_zero",
                extra={
                    "variance": variance,
                    "threshold": _VARIANCE_EPSILON,
                    "reason": "near_constant_signal",
                },
            )
        
        return estimate_derivatives(values, dt, effective_order, self._method)

    def _classify_trend(self, slope: float) -> str:
        if slope > self._trend_threshold:
            return "up"
        if slope < -self._trend_threshold:
            return "down"
        return "stable"

    def _confidence_from_stability(self, stability: float) -> float:
        instability = min(stability, 0.7)
        c = max(self._min_confidence, 1.0 - instability)
        return min(c, self._max_confidence)

    def _compute_variance(self, values: List[float]) -> float:
        """Compute population variance of values (CRIT-4).
        
        Args:
            values: Time series values
        
        Returns:
            Population variance, or 0.0 if insufficient data
        """
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _fallback(self, values: List[float]) -> PredictionResult:
        tail = values[-min(3, len(values)):]
        predicted = sum(tail) / len(tail)
        logger.debug("taylor_fallback", extra={"n": len(values)})
        return PredictionResult(
            predicted_value=predicted,
            confidence=max(self._min_confidence, min(0.5, len(values) / 10.0)),
            trend="stable",
            metadata={
                "order": 0,
                "derivatives": {"f_t": values[-1] if values else 0.0,
                                "f_prime": 0.0, "f_double_prime": 0.0,
                                "f_triple_prime": 0.0},
                "dt": 1.0, "horizon_steps": self._horizon,
                "fallback": "insufficient_data",
                "clamped": False, "diagnostic": None,
            },
        )
