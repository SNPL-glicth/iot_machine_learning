"""Statistical prediction engine — EMA/WMA-based forecasting.

Provides a complementary perspective to Taylor's derivative-based
approach.  Uses exponential moving average (EMA) with optional
trend correction (double exponential smoothing / Holt's method).

Design:
    - Auto-tuning of α and β via grid search.
    - Persistence of optimal parameters per series.
    - Re-optimization after 50 predictions if MAE improves > 5%.
    - Configurable smoothing factor α and trend factor β.
    - Exposes diagnostic metadata for the cognitive orchestrator.
    - Noise-resistant by construction (EMA is a low-pass filter).

FASE 3: Added adaptive parameter optimization and persistence.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from core.parameters.numerical_constants import EPSILON
from core.statistical.statistical_validation import StationarityValidator, StationarityTestResult

from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)
from iot_machine_learning.infrastructure.ml.cognitive.hyperparameters import (
    HyperparameterAdaptor,
)

logger = logging.getLogger(__name__)


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
    
    DEPRECATED: Use _holt_stable() to prevent trend explosion.
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


# MATH-CRIT-3: Configurable constants (no magic numbers)
_DEFAULT_MAX_TREND_RATIO: float = 0.5
_MIN_LEVEL_FOR_DAMPING: float = EPSILON.GRADIENT


def _holt_stable(
    values: List[float],
    alpha: float,
    beta: float,
    max_trend_ratio: float = _DEFAULT_MAX_TREND_RATIO,
) -> tuple[float, float]:
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
    
    Applies SRP: Trend stabilization is independent concern.
    Applies OCP: Can be used as drop-in replacement for _holt().
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
                # Apply damping: scale trend to max_trend_ratio
                damping_factor = max_trend_ratio / trend_ratio
                trend = trend * damping_factor
    
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
    """EMA/Holt-based prediction engine with adaptive parameters.

    Attributes:
        _alpha: EMA smoothing factor (higher = more reactive).
        _beta: Trend smoothing factor for Holt's method.
        _horizon: Steps ahead to predict.
        _series_id: Series identifier for parameter persistence.
        _prediction_history: Recent predictions for re-optimization.
        _prediction_count: Count of predictions since last optimization.
        _current_mae: Current MAE from loaded params.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        horizon: int = 1,
        series_id: Optional[str] = None,
        enable_optimization: bool = True,
        hyperparameter_adaptor: Optional[HyperparameterAdaptor] = None,
        max_trend_ratio: float = _DEFAULT_MAX_TREND_RATIO,
        stationarity_validator: Optional[StationarityValidator] = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if max_trend_ratio <= 0.0:
            raise ValueError(f"max_trend_ratio must be > 0, got {max_trend_ratio}")

        self._series_id = series_id
        self._enable_optimization = enable_optimization
        self._prediction_history: Deque[float] = deque(maxlen=100)
        self._prediction_count = 0
        self._needs_reoptimization = False
        self._current_mae = 999.0
        self._stationarity_validator = stationarity_validator
        self._stationarity_result: Optional[StationarityTestResult] = None

        # IMP-4c: HyperparameterAdaptor is the sole source of truth.
        # StatisticalParamsRepository coexistence was dropped.
        self._hyperparams = hyperparameter_adaptor

        self._alpha = alpha
        self._beta = beta
        self._horizon = horizon
        self._max_trend_ratio = max_trend_ratio  # MATH-CRIT-3

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
        """Predict next value using EMA or Holt's method.

        P4: Online alpha micro-adjustment happens in ``record_actual()``.
        When data is non-stationary, falls back to EMA-only.

        Args:
            values: Time-series window.
            timestamps: Ignored for this engine (kept for interface parity).

        Returns:
            PredictionResult with level, trend_component, alpha, beta metadata.

        Raises:
            ValueError: If values is empty.
        """
        if not values:
            raise ValueError("values cannot be empty")

        # IMP-4c: refresh per-series hyperparameters before each predict().
        self._load_hyperparams(window_size=len(values))

        n = len(values)
        if not self.can_handle(n):
            return self._fallback(values)

        # Validate stationarity if validator provided
        if self._stationarity_validator is not None and len(values) >= self._stationarity_validator.min_samples:
            data_array = np.array(values)
            self._stationarity_result = self._stationarity_validator.validate(data_array)
            logger.info(
                "statistical_stationarity_validation",
                extra={
                    "is_stationary": self._stationarity_result.is_stationary,
                    "stationarity_type": self._stationarity_result.stationarity_type.value,
                    "recommendation": self._stationarity_result.recommendation,
                    "adf_p": self._stationarity_result.adf_p_value,
                },
            )

            # Fallback to EMA if non-stationary
            if not self._stationarity_result.is_stationary:
                logger.warning(
                    "statistical_non_stationary_fallback",
                    extra={"recommendation": "use_ema"},
                )
                # Use EMA instead of Holt's method
                ema_series = _ema(values, self._alpha)
                predicted = ema_series[-1]
                residual_std = _compute_residual_std(values, ema_series)
                mean_abs = abs(sum(values) / n) if n > 0 else 1.0
                noise_ratio = residual_std / (mean_abs + EPSILON.DIVISION)
                confidence = max(0.2, min(0.95, 1.0 - noise_ratio))
                trend_dir = "stable"
                stability = min(1.0, noise_ratio)
                metadata = {
                    "level": predicted,
                    "trend_component": 0.0,
                    "alpha": self._alpha,
                    "beta": 0.0,
                    "residual_std": round(residual_std, 6),
                    "horizon_steps": self._horizon,
                    "fallback": "ema_instead_of_holt",
                    "diagnostic": {
                        "stability_indicator": round(stability, 4),
                        "local_fit_error": round(residual_std, 6),
                        "method": "ema_only",
                    },
                }
                return PredictionResult(
                    predicted_value=predicted,
                    confidence=confidence,
                    trend=trend_dir,
                    metadata=metadata,
                )

        # Holt's double exponential smoothing with stability (MATH-CRIT-3)
        level, trend = _holt_stable(values, self._alpha, self._beta, self._max_trend_ratio)
        predicted = level + trend * self._horizon

        # EMA for residual analysis
        ema_series = _ema(values, self._alpha)
        residual_std = _compute_residual_std(values, ema_series)

        # Confidence from residual stability
        mean_abs = abs(sum(values) / n) if n > 0 else 1.0
        noise_ratio = residual_std / (mean_abs + EPSILON.DIVISION)
        confidence = max(0.2, min(0.95, 1.0 - noise_ratio))

        # Trend classification
        if abs(trend) < max(residual_std * 0.1, EPSILON.COMPARISON):
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
    
    def record_actual(self, predicted: float, actual: float) -> None:
        """Record actual value for online alpha adjustment + re-optimization.

        P4: Two-phase learning:
        1. Online alpha micro-adjustment every call (fast, cheap).
        2. Deferred full re-optimization via ``optimize()`` every 20 calls.

        Args:
            predicted: Predicted value
            actual: Actual observed value
        """
        if not self._enable_optimization:
            return

        error = abs(predicted - actual)
        self._prediction_history.append(actual)
        self._prediction_count += 1

        # P4: Online alpha micro-adjustment
        self._online_adjust_alpha(error, actual)

        # P4: Deferred full re-optimization threshold lowered from 50 → 20
        if self._prediction_count >= 20 and len(self._prediction_history) >= 20:
            self._needs_reoptimization = True

    def _online_adjust_alpha(self, error: float, actual: float) -> None:
        """Micro-adjust alpha based on recent prediction error (P4).

        Logic:
        - Large error → increase alpha (more reactive to new data).
        - Small error → decrease alpha (more smoothing, stable).
        - Learning rate = 0.01, clamped to [0.05, 0.95].
        """
        scale = max(abs(actual), 1.0)
        normalized_error = error / scale
        lr = 0.01
        # If error > 10% of scale, nudge alpha up; if < 10%, nudge down
        delta = lr * (normalized_error - 0.1)
        self._alpha = max(0.05, min(0.95, self._alpha + delta))
        logger.debug(
            "statistical_alpha_adjusted",
            extra={
                "alpha": round(self._alpha, 4),
                "delta": round(delta, 6),
                "normalized_error": round(normalized_error, 4),
            },
        )

    def optimize(self) -> None:
        """Trigger deferred re-optimization if threshold was reached.

        Safe to call from orchestrator between requests — fully
        externalizes the mutable-state write that was inside
        ``record_actual()``.
        """
        if not self._needs_reoptimization:
            return
        self._reoptimize()
        self._needs_reoptimization = False
        self._prediction_count = 0
    
    def _load_hyperparams(self, window_size: int) -> None:
        """Load adaptive alpha/beta from the HyperparameterAdaptor (IMP-4c).

        Redis is the sole source of truth. When the adaptor is inert or
        the series has no stored params, the engine keeps the defaults
        configured at construction time.
        """
        if self._hyperparams is None or not self._series_id:
            return
        params = self._hyperparams.load(self._series_id, self.name)
        if not params:
            return
        alpha_raw = params.get("alpha")
        beta_raw = params.get("beta")
        mae_raw = params.get("mae")
        if alpha_raw is not None:
            try:
                a = float(alpha_raw)
                if 0.0 < a <= 1.0:
                    self._alpha = a
            except (TypeError, ValueError):
                pass
        if beta_raw is not None:
            try:
                b = float(beta_raw)
                if 0.0 <= b <= 1.0:
                    self._beta = b
            except (TypeError, ValueError):
                pass
        if mae_raw is not None:
            try:
                self._current_mae = float(mae_raw)
            except (TypeError, ValueError):
                pass
        logger.debug(
            "hyperparams_loaded series=%s engine=%s alpha=%.4f beta=%.4f window=%d",
            self._series_id,
            self.name,
            self._alpha,
            self._beta,
            window_size,
        )

    def _reoptimize(self) -> None:
        """Re-optimize parameters if MAE improves > 5%% (IMP-4c).

        Persists new params via the HyperparameterAdaptor (Redis). When
        no adaptor is configured the re-optimization is skipped — the
        engine simply keeps its current in-memory params.
        """
        if not self._series_id:
            return

        try:
            from iot_machine_learning.infrastructure.ml.engines.statistical.param_optimizer import (
                StatisticalParamOptimizer,
            )

            optimizer = StatisticalParamOptimizer()
            values = list(self._prediction_history)

            new_alpha, new_beta, new_mae = optimizer.optimize(values)

            improvement = (self._current_mae - new_mae) / self._current_mae
            if improvement > 0.05:
                self._alpha = new_alpha
                self._beta = new_beta
                self._current_mae = new_mae

                if self._hyperparams is not None:
                    self._hyperparams.save(
                        self._series_id,
                        self.name,
                        {
                            "alpha": new_alpha,
                            "beta": new_beta,
                            "mae": new_mae,
                        },
                    )

                logger.info(
                    "statistical_params_reoptimized",
                    extra={
                        "series_id": self._series_id,
                        "new_alpha": new_alpha,
                        "new_beta": new_beta,
                        "new_mae": round(new_mae, 4),
                        "improvement_pct": round(improvement * 100, 2),
                    },
                )
        except Exception as exc:
            logger.warning(
                "statistical_reoptimization_failed",
                extra={"series_id": self._series_id, "error": str(exc)},
            )

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
