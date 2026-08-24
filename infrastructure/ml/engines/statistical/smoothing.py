"""Smoothing algorithms: EMA and Holt's double exponential smoothing."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from core.parameters.numerical_constants import EPSILON


def ema(values: List[float], alpha: float) -> List[float]:
    """Compute exponential moving average series."""
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return result


# MATH-CRIT-3: Configurable constants
_DEFAULT_MAX_TREND_RATIO: float = 0.5
_MIN_LEVEL_FOR_DAMPING: float = EPSILON.GRADIENT


def holt_stable(
    values: List[float],
    alpha: float,
    beta: float,
    max_trend_ratio: float = _DEFAULT_MAX_TREND_RATIO,
) -> Tuple[float, float]:
    """Double exponential smoothing with trend damping (MATH-CRIT-3).
    
    Prevents trend explosion in non-stationary data by applying damping
    when trend grows too large relative to level.
    
    Args:
        values: Time series values.
        alpha: Level smoothing factor (0, 1].
        beta: Trend smoothing factor [0, 1].
        max_trend_ratio: Maximum allowed |trend/level| before damping.
    
    Returns:
        Tuple of (level, trend) at the last point.
    """
    if len(values) < 2:
        return (values[0] if values else 0.0), 0.0
    
    level = values[0]
    trend = values[1] - values[0]
    
    for v in values[1:]:
        prev_level = level
        level = alpha * v + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
        
        # MATH-CRIT-3: Stability check - damp trend if too large
        if abs(level) > _MIN_LEVEL_FOR_DAMPING:
            trend_ratio = abs(trend / level)
            if trend_ratio > max_trend_ratio:
                damping_factor = max_trend_ratio / trend_ratio
                trend = trend * damping_factor
    
    return level, trend


def compute_residual_std(values: List[float], ema_series: List[float]) -> float:
    """Standard deviation of residuals (values - EMA)."""
    n = len(values)
    if n < 2:
        return 0.0
    residuals = [values[i] - ema_series[i] for i in range(n)]
    mu = sum(residuals) / n
    var = sum((r - mu) ** 2 for r in residuals) / n
    return math.sqrt(var)


def compute_confidence(values: List[float], residual_std: float) -> float:
    """Compute confidence from residual stability."""
    n = len(values)
    mean_abs = abs(sum(values) / n) if n > 0 else 1.0
    noise_ratio = residual_std / (mean_abs + EPSILON.DIVISION)
    return max(0.2, min(0.95, 1.0 - noise_ratio))


def classify_trend(trend: float, residual_std: float) -> str:
    """Classify trend direction."""
    if abs(trend) < max(residual_std * 0.1, EPSILON.COMPARISON):
        return "stable"
    elif trend > 0:
        return "up"
    else:
        return "down"