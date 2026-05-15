"""Kalman engine helpers — pure functions extracted to keep engine.py <180 lines."""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import List, Optional, Tuple

from core.parameters.numerical_constants import CONFIDENCE

from .kalman_cv_math import (
    _compute_process_noise_covariance,
    predict_cv,
)

logger = logging.getLogger(__name__)


def sanitize_inputs(
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


def estimate_dt(
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


def detect_gap(
    timestamps: Optional[List[float]],
    dt: float,
) -> bool:
    """Return True if any inter-sample gap exceeds 5×dt."""
    if timestamps is None or len(timestamps) < 2:
        return False
    gaps = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]
    if gaps and max(gaps) > 5.0 * dt:
        logger.warning(
            "kalman_gap_detected",
            extra={
                "engine": "kalman",
                "max_gap": max(gaps),
                "threshold": 5.0 * dt,
                "dt": dt,
            },
        )
        return True
    return False


def compute_confidence(
    P_pos: float,
    clean_values: List[float],
    n: int,
    warmup_size: int,
) -> Tuple[float, float]:
    """Return (confidence, scale) with warmup degradation."""
    mean_val = sum(clean_values) / n
    mean_abs = abs(mean_val)
    std_val = math.sqrt(
        sum((v - mean_val) ** 2 for v in clean_values) / n
    )
    value_range = max(clean_values) - min(clean_values)
    scale = max(
        mean_abs,
        std_val,
        value_range * 0.1,
        0.01,
    )
    k = 0.5 / scale

    confidence = CONFIDENCE.MAX_CONFIDENCE / (1.0 + k * math.sqrt(P_pos))
    confidence = max(
        CONFIDENCE.MIN_CONFIDENCE,
        min(CONFIDENCE.MAX_CONFIDENCE, confidence),
    )

    # Warmup degradation if n < warmup_size * 2
    if n < warmup_size * 2:
        progress = n / (warmup_size * 2)
        confidence *= progress ** 2
        confidence = max(CONFIDENCE.MIN_CONFIDENCE, confidence)

    return confidence, scale


def classify_trend(v_hat: float, scale: float) -> str:
    """Classify trend from velocity estimate (scale-relative)."""
    threshold_rel = 0.01 * scale
    if v_hat > threshold_rel:
        return "up"
    elif v_hat < -threshold_rel:
        return "down"
    return "stable"


def build_metadata(
    state,
    predicted: float,
    scale: float,
    gap_detected: bool,
    horizon: int,
    dt: float,
    confidence_interval: tuple,
) -> dict:
    """Construct Kalman metadata dict."""
    P_pos = float(state.P[0, 0])
    sigma_pos = math.sqrt(P_pos)
    return {
        "x_hat": round(state.x, 6),
        "v_hat": round(state.v, 6),
        "P_pos": round(P_pos, 6),
        "P_vel": round(float(state.P[1, 1]), 6),
        "R": round(state.R, 6),
        "Q_vel": round(float(state.Q[1, 1]), 6),
        "Q_adaptive": len(state._innovations) >= 3,
        "innovation_window": len(state._innovations),
        "warmup_complete": state.initialized,
        "horizon": horizon,
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


def _fallback(
    clean_values: List[float], n: int, warmup_size: int,
) -> "PredictionResult":
    """Return fallback PredictionResult when insufficient data."""
    from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult

    logger.warning(
        "kalman_insufficient_data",
        extra={"engine": "kalman", "n_points": n, "required": warmup_size},
    )
    fallback_value = (
        sum(clean_values) / max(n, 1) if clean_values else 0.0
    )
    return PredictionResult(
        predicted_value=fallback_value,
        confidence=0.0,
        trend="stable",
        metadata={"reason": "insufficient_data", "engine": "kalman"},
    )


def _predict_horizon(state, dt: float, horizon: int):
    """Predict horizon steps ahead with correct process noise."""
    if horizon == 1:
        return predict_cv(state, dt)

    total_dt = dt * horizon
    q_intensity = float(state.Q[1, 1] / (dt if dt > 0 else 1.0))
    Q_total = _compute_process_noise_covariance(total_dt, q_intensity)
    state_total = dataclasses.replace(state, Q=Q_total)
    return predict_cv(state_total, total_dt)
