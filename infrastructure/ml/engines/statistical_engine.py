"""Statistical prediction engine — EMA/WMA-based forecasting.

Provides a complementary perspective to Taylor's derivative-based
approach.  Uses exponential moving average (EMA) with optional
trend correction (double exponential smoothing / Holt's method).

Design:
    - Stateless per call (no warmup buffer).
    - Configurable smoothing factor α and trend factor β.
    - Exposes diagnostic metadata for the cognitive orchestrator.
    - Noise-resistant by construction (EMA is a low-pass filter).

Pure computation — no I/O, no persistence.
"""

from __future__ import annotations

import math
from typing import List, Optional

from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)


def _ema(values: List[float], alpha: float) -> List[float]:
    """Compute exponential moving average series."""
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return result


def _holt(
    values: List[float],
    alpha: float,
    beta: float,
) -> tuple[float, float]:
    """Double exponential smoothing (Holt's method).

    Returns (level, trend) at the last point.
    """
    if len(values) < 2:
        return (values[0] if values else 0.0), 0.0

    level = values[0]
    trend = values[1] - values[0]

    for v in values[1:]:
        prev_level = level
        level = alpha * v + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend

    return level, trend


def _compute_residual_std(
    values: List[float],
    ema_series: List[float],
) -> float:
    """Standard deviation of residuals (values - EMA)."""
    n = len(values)
    if n < 2:
        return 0.0
    residuals = [values[i] - ema_series[i] for i in range(n)]
    mu = sum(residuals) / n
    var = sum((r - mu) ** 2 for r in residuals) / n
    return math.sqrt(var)


class StatisticalPredictionEngine(PredictionEngine):
    """EMA/Holt-based prediction engine.

    Attributes:
        _alpha: EMA smoothing factor (higher = more reactive).
        _beta: Trend smoothing factor for Holt's method.
        _horizon: Steps ahead to predict.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        horizon: int = 1,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self._alpha = alpha
        self._beta = beta
        self._horizon = horizon

    @property
    def name(self) -> str:
        return "statistical_ema_holt"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= 3

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        if not values:
            raise ValueError("values cannot be empty")

        n = len(values)
        if not self.can_handle(n):
            return self._fallback(values)

        # Holt's double exponential smoothing
        level, trend = _holt(values, self._alpha, self._beta)
        predicted = level + trend * self._horizon

        # EMA for residual analysis
        ema_series = _ema(values, self._alpha)
        residual_std = _compute_residual_std(values, ema_series)

        # Confidence from residual stability
        mean_abs = abs(sum(values) / n) if n > 0 else 1.0
        noise_ratio = residual_std / (mean_abs + 1e-9)
        confidence = max(0.2, min(0.95, 1.0 - noise_ratio))

        # Trend classification
        if abs(trend) < max(residual_std * 0.1, 1e-9):
            trend_dir = "stable"
        elif trend > 0:
            trend_dir = "up"
        else:
            trend_dir = "down"

        # Stability indicator
        stability = min(1.0, noise_ratio)

        metadata: dict = {
            "level": round(level, 6),
            "trend_component": round(trend, 6),
            "alpha": self._alpha,
            "beta": self._beta,
            "residual_std": round(residual_std, 6),
            "horizon_steps": self._horizon,
            "fallback": None,
            "diagnostic": {
                "stability_indicator": round(stability, 4),
                "local_fit_error": round(residual_std, 6),
                "method": "ema_holt",
            },
        }

        return PredictionResult(
            predicted_value=predicted,
            confidence=confidence,
            trend=trend_dir,
            metadata=metadata,
        )

    def supports_uncertainty(self) -> bool:
        return False

    def _fallback(self, values: List[float]) -> PredictionResult:
        tail = values[-min(3, len(values)):]
        predicted = sum(tail) / len(tail)
        return PredictionResult(
            predicted_value=predicted,
            confidence=0.3,
            trend="stable",
            metadata={
                "fallback": "insufficient_data",
                "diagnostic": None,
            },
        )
