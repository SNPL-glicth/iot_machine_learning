"""Kalman prediction engine — 2D constant-velocity model.

Implements PredictionEngine using pure 2D Kalman math from kalman_cv_math.
Designed for NOISY/VOLATILE regimes where separating measurement noise
from process dynamics improves prediction accuracy.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import List, Optional


from iot_machine_learning.infrastructure.ml.engines.core.factory import (
    register_engine,
)
from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)

from .kalman_cv_math import initialize_cv_state, adaptive_cv_update
from .engine_helpers import (
    _fallback,
    _predict_horizon,
    build_metadata,
    classify_trend,
    compute_confidence,
    detect_gap,
    estimate_dt,
    sanitize_inputs,
)

logger = logging.getLogger(__name__)

_MIN_WARMUP_SIZE: int = 3
_DEFAULT_WARMUP_SIZE: int = 5


@register_engine("kalman")
class KalmanPredictionEngine(PredictionEngine):
    """2D Constant-Velocity Kalman prediction engine.

    Separates measurement noise (R) from process dynamics (Q) to produce
    robust predictions in noisy environments.  Provides real confidence
    intervals via Kalman covariance P[0,0].

    Args:
        warmup_size: Points required for initial calibration (min 3).
        horizon: Steps ahead to predict.
        q_scale: Initial process noise intensity.
    """

    def __init__(
        self,
        warmup_size: int = _DEFAULT_WARMUP_SIZE,
        horizon: int = 1,
        q_scale: float = 1.0,
    ) -> None:
        if warmup_size < _MIN_WARMUP_SIZE:
            logger.warning(
                "kalman_warmup_clamped",
                extra={"requested": warmup_size, "minimum": _MIN_WARMUP_SIZE},
            )
            warmup_size = _MIN_WARMUP_SIZE
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if q_scale <= 0:
            raise ValueError(f"q_scale must be > 0, got {q_scale}")

        self._warmup_size = warmup_size
        self._horizon = horizon
        self._q_scale = q_scale
        self._error_history: deque = deque(maxlen=50)

    @property
    def name(self) -> str:
        return "kalman"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= self._warmup_size

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        """Predict next value using 2D CV Kalman filter."""
        clean_values, clean_timestamps = sanitize_inputs(values, timestamps)
        n = len(clean_values)

        if not self.can_handle(n):
            return _fallback(clean_values, n, self._warmup_size)

        dt = estimate_dt(clean_timestamps, n)
        gap_detected = detect_gap(clean_timestamps, dt)

        warmup = clean_values[: self._warmup_size]
        state = initialize_cv_state(
            warmup_values=warmup, dt=dt, Q_scale=self._q_scale,
        )

        for measurement in clean_values[self._warmup_size :]:
            state = adaptive_cv_update(state, measurement)

        state = _predict_horizon(state, dt, self._horizon)
        predicted = state.x
        P_pos = float(state.P[0, 0])

        confidence, scale = compute_confidence(
            P_pos, clean_values, n, self._warmup_size,
        )
        trend_dir = classify_trend(state.v, scale)

        sigma_pos = math.sqrt(P_pos)
        confidence_interval = (predicted - 2.0 * sigma_pos,
                               predicted + 2.0 * sigma_pos)

        metadata = build_metadata(
            state, predicted, scale, gap_detected,
            self._horizon, dt, confidence_interval,
        )

        logger.debug(
            "kalman_predict",
            extra={
                "engine": "kalman", "n_points": n,
                "confidence": round(confidence, 4),
                "v_hat": round(state.v, 6),
                "P_pos": round(P_pos, 6),
                "warmup_complete": state.initialized,
                "gap_detected": gap_detected,
            },
        )
        return PredictionResult(
            predicted_value=predicted,
            confidence=confidence,
            trend=trend_dir,
            metadata=metadata,
        )

    def supports_uncertainty(self) -> bool:
        return True

    def record_actual(self, predicted: float, actual: float) -> None:
        """Record actual value for MAE tracking.

        Validates finiteness before accumulation.  Non-finite values
        are silently skipped with a warning log.
        """
        if not math.isfinite(predicted) or not math.isfinite(actual):
            logger.warning(
                "kalman_record_actual_non_finite",
                extra={"predicted": predicted, "actual": actual},
            )
            return
        error = abs(predicted - actual)
        self._error_history.append(error)
        logger.debug(
            "kalman_record_actual",
            extra={
                "predicted": round(predicted, 4),
                "actual": round(actual, 4),
                "error": round(error, 4),
                "history_size": len(self._error_history),
            },
        )

    def recent_mae(self) -> float:
        """Return mean absolute error over recent predictions."""
        if not self._error_history:
            return 0.0
        return sum(self._error_history) / len(self._error_history)

