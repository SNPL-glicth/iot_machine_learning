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

from core.parameters.numerical_constants import CONFIDENCE, EPSILON

from iot_machine_learning.infrastructure.ml.engines.core.factory import (
    register_engine,
)
from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)

from .kalman_cv_math import initialize_cv_state, predict_cv, update_cv

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
        """Predict next value using 2D CV Kalman filter.

        Flow:
        1. Sanitize inputs (drop NaN/inf silently).
        2. Check minimum points.
        3. Estimate dt from timestamps.
        4. Detect temporal gaps.
        5. Initialize Kalman state from warmup.
        6. Run sequential predict+update over remaining points.
        7. Predict horizon steps ahead.
        8. Compute scale-normalized confidence from P[0,0].
        9. Degrade confidence during extended warmup.
        10. Classify trend from velocity estimate.
        11. Build metadata with confidence interval.
        """
        # a) Sanitize
        clean_values, clean_timestamps = self._sanitize_inputs(
            values, timestamps
        )
        n = len(clean_values)

        # b) Check can_handle
        if not self.can_handle(n):
            logger.warning(
                "kalman_insufficient_data",
                extra={
                    "engine": "kalman",
                    "n_points": n,
                    "required": self._warmup_size,
                },
            )
            fallback_value = (
                sum(clean_values) / max(n, 1) if clean_values else 0.0
            )
            return PredictionResult(
                predicted_value=fallback_value,
                confidence=0.0,
                trend="stable",
                metadata={
                    "reason": "insufficient_data",
                    "engine": "kalman",
                },
            )

        # c) Estimate dt
        dt = self._estimate_dt(clean_timestamps, n)

        # d) Gap detection
        gap_detected = False
        if clean_timestamps is not None and len(clean_timestamps) >= 2:
            gaps = [
                clean_timestamps[i] - clean_timestamps[i - 1]
                for i in range(1, len(clean_timestamps))
            ]
            if gaps and max(gaps) > 5.0 * dt:
                gap_detected = True
                logger.warning(
                    "kalman_gap_detected",
                    extra={
                        "engine": "kalman",
                        "max_gap": max(gaps),
                        "threshold": 5.0 * dt,
                        "dt": dt,
                    },
                )

        # e) Initialize state from warmup
        warmup = clean_values[: self._warmup_size]
        state = initialize_cv_state(
            warmup_values=warmup,
            dt=dt,
            Q_scale=self._q_scale,
        )

        # f) Sequential update over all points post-warmup
        for measurement in clean_values[self._warmup_size :]:
            state = predict_cv(state, dt)
            state = update_cv(state, measurement)

        # g) Predict horizon h
        for _ in range(self._horizon):
            state = predict_cv(state, dt)

        predicted = state.x

        # h) Confidence with scale correction
        mean_val = sum(clean_values) / n
        mean_abs = abs(mean_val)
        std_val = math.sqrt(
            sum((v - mean_val) ** 2 for v in clean_values) / n
        )
        scale = max(mean_abs, std_val, float(EPSILON.DIVISION))
        k = 0.5 / scale

        P_pos = float(state.P[0, 0])
        confidence = CONFIDENCE.MAX_CONFIDENCE / (
            1.0 + k * math.sqrt(P_pos)
        )
        confidence = max(
            CONFIDENCE.MIN_CONFIDENCE,
            min(CONFIDENCE.MAX_CONFIDENCE, confidence),
        )

        # i) Warmup degradation if n < warmup_size * 2
        if n < self._warmup_size * 2:
            progress = n / (self._warmup_size * 2)
            confidence *= progress ** 2
            confidence = max(CONFIDENCE.MIN_CONFIDENCE, confidence)

        # j) Trend classification (scale-relative)
        threshold_rel = 0.01 * scale
        v_hat = state.v
        if v_hat > threshold_rel:
            trend_dir = "up"
        elif v_hat < -threshold_rel:
            trend_dir = "down"
        else:
            trend_dir = "stable"

        # k) Metadata
        sigma_pos = math.sqrt(P_pos)
        confidence_interval = (
            predicted - 2.0 * sigma_pos,
            predicted + 2.0 * sigma_pos,
        )

        metadata: dict = {
            "x_hat": round(state.x, 6),
            "v_hat": round(state.v, 6),
            "P_pos": round(P_pos, 6),
            "P_vel": round(float(state.P[1, 1]), 6),
            "R": round(state.R, 6),
            "Q_vel": round(float(state.Q[1, 1]), 6),
            "warmup_complete": state.initialized,
            "horizon": self._horizon,
            "dt": dt,
            "gap_detected": gap_detected,
            "confidence_interval": (
                round(confidence_interval[0], 6),
                round(confidence_interval[1], 6),
            ),
            "diagnostic": {
                "stability_indicator": round(sigma_pos / scale, 4),
                "local_fit_error": round(sigma_pos, 6),
            },
        }

        logger.debug(
            "kalman_predict",
            extra={
                "engine": "kalman",
                "n_points": n,
                "confidence": round(confidence, 4),
                "v_hat": round(v_hat, 6),
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

    # -- private helpers --------------------------------------------------

    @staticmethod
    def _sanitize_inputs(
        values: List[float],
        timestamps: Optional[List[float]],
    ) -> tuple[List[float], Optional[List[float]]]:
        """Filter NaN and inf silently."""
        ts_iter = timestamps if timestamps is not None else [None] * len(values)
        clean = [
            (v, t)
            for v, t in zip(values, ts_iter)
            if v is not None
            and not (v != v)  # NaN check
            and abs(v) != float("inf")
        ]
        if not clean:
            return [], None
        c_values, c_timestamps_raw = zip(*clean)
        c_timestamps = (
            list(c_timestamps_raw) if timestamps is not None else None
        )
        return list(c_values), c_timestamps

    @staticmethod
    def _estimate_dt(
        timestamps: Optional[List[float]],
        n_points: int,
    ) -> float:
        """Estimate uniform time step from median of diffs."""
        if timestamps is None or len(timestamps) < 2:
            return 1.0
        diffs = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
        ]
        if not diffs:
            return 1.0
        sorted_diffs = sorted(diffs)
        mid = len(sorted_diffs) // 2
        if len(sorted_diffs) % 2 == 1:
            return float(sorted_diffs[mid])
        return float((sorted_diffs[mid - 1] + sorted_diffs[mid]) / 2.0)
