"""Taylor prediction engine — orchestrator.

Delegates all math to the ``taylor/`` package and focuses on:
1. Input validation
2. Order negotiation
3. Clamping raw predictions to observed range
4. Trend classification from first derivative
5. Confidence estimation from stability indicator
6. Assembling metadata with TaylorDiagnostic
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
        dt = compute_dt(timestamps)

        if not self.can_handle(n):
            return self._fallback(values)

        # CRIT-4: Check variance to prevent overfitting to noise
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

        coeffs = estimate_derivatives(values, dt, effective_order, self._method)
        if coeffs.estimated_order == 0:
            return self._fallback(values)

        h = float(self._horizon) * dt
        predicted_raw = project(coeffs, h, coeffs.estimated_order)
        predicted, clamped = clamp_prediction(
            predicted_raw, values, margin_pct=self._clamp_margin_pct,
        )
        trend = self._classify_trend(coeffs.local_slope)
        diag = compute_diagnostic(coeffs, values, dt)
        confidence = self._confidence_from_stability(diag.stability_indicator)

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
            "variance_check": variance,  # CRIT-4: Include for debugging
            "variance_threshold": _VARIANCE_EPSILON,
        }
        logger.debug(
            "taylor_prediction",
            extra={
                "n_points": n, "effective_order": coeffs.estimated_order,
                "method": coeffs.method, "predicted": predicted,
                "clamped": clamped, "confidence": confidence,
                "stability": diag.stability_indicator,
                "variance": variance,
            },
        )
        return PredictionResult(
            predicted_value=predicted, confidence=confidence,
            trend=trend, metadata=metadata,
        )

    def supports_uncertainty(self) -> bool:
        return False

    # -- private helpers --------------------------------------------------

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
