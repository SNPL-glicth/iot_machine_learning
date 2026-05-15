"""Stateless feature builder for LightGBMPredictionEngine (P5).

All functions are pure: no mutable state, no I/O, no side effects.
Engine state (trained model) lives exclusively in the engine.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


def build_feature_vector(
    values: List[float],
    timestamps: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Extract a feature dictionary from a time-series window.

    Features:
        - lag_1, lag_2, lag_3: last 3 raw values
        - ma_3, ma_5: rolling mean of last 3 and 5 values
        - std_5: rolling std of last 5 values
        - delta_1, delta_2: first and second differences
        - trend: linear slope via least-squares on last 10 points
        - hour_sin, hour_cos: cyclic hour-of-day (if timestamps provided)
    """
    n = len(values)
    if n == 0:
        return {}

    feats: Dict[str, float] = {}

    # Lag features
    feats["lag_1"] = values[-1]
    feats["lag_2"] = values[-2] if n >= 2 else values[-1]
    feats["lag_3"] = values[-3] if n >= 3 else feats["lag_2"]

    # Rolling means
    feats["ma_3"] = _mean(values[-3:]) if n >= 3 else _mean(values)
    feats["ma_5"] = _mean(values[-5:]) if n >= 5 else _mean(values)

    # Rolling std
    feats["std_5"] = _stdev(values[-5:]) if n >= 5 else 0.0

    # Deltas
    feats["delta_1"] = values[-1] - values[-2] if n >= 2 else 0.0
    feats["delta_2"] = (
        (values[-1] - values[-2]) - (values[-2] - values[-3]) if n >= 3 else 0.0
    )

    # Linear trend on last 10 points
    trend_window = values[-10:]
    feats["trend"] = _linear_slope(trend_window) if len(trend_window) >= 2 else 0.0

    # Hour-of-day cyclic features (if timestamps available)
    if timestamps is not None and len(timestamps) > 0:
        ts_last = timestamps[-1]
        hour = _hour_of_day(ts_last)
        if hour is not None:
            rad = 2.0 * math.pi * hour / 24.0
            feats["hour_sin"] = math.sin(rad)
            feats["hour_cos"] = math.cos(rad)
        else:
            feats["hour_sin"] = 0.0
            feats["hour_cos"] = 0.0
    else:
        feats["hour_sin"] = 0.0
        feats["hour_cos"] = 0.0

    return feats


def build_training_matrix(
    values: List[float],
    timestamps: Optional[List[float]] = None,
    min_points: int = 80,
) -> Tuple[List[List[float]], List[str], List[float]]:
    """Build supervised training matrix from a long series.

    Uses a sliding window of size ``min_points`` to create (X, y) pairs.
    Each row in X corresponds to features built from window[:-1] and y is window[-1].

    Returns:
        Tuple of (X_matrix, feature_names, y_vector).
    """
    if len(values) < min_points + 1:
        return [], [], []

    X: List[List[float]] = []
    y: List[float] = []
    feature_names: List[str] = []

    for i in range(min_points, len(values)):
        window = values[:i]
        ts_window = timestamps[:i] if timestamps is not None else None
        feats = build_feature_vector(window, ts_window)
        if not feature_names:
            feature_names = list(feats.keys())
        row = [feats.get(k, 0.0) for k in feature_names]
        X.append(row)
        y.append(values[i])

    return X, feature_names, y


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------

def _mean(seq: List[float]) -> float:
    if not seq:
        return 0.0
    return sum(seq) / len(seq)


def _stdev(seq: List[float]) -> float:
    if len(seq) < 2:
        return 0.0
    m = _mean(seq)
    var = sum((x - m) ** 2 for x in seq) / (len(seq) - 1)
    return math.sqrt(var)


def _linear_slope(seq: List[float]) -> float:
    """Least-squares slope assuming uniform x = 0, 1, 2, ..."""
    n = len(seq)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = _mean(seq)
    num = sum((i - x_mean) * (seq[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _hour_of_day(ts: float) -> Optional[int]:
    """Extract hour-of-day from a Unix timestamp."""
    try:
        import datetime

        dt = datetime.datetime.utcfromtimestamp(ts)
        return dt.hour
    except (ValueError, OSError, OverflowError):
        return None
