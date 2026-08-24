"""Statistical prediction engine — EMA/Holt-based forecasting.

Composed of:
- Smoothing algorithms (EMA, Holt)
- Stationarity validation
- Hyperparameter loading/persistence
- Parameter re-optimization
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from core.parameters.numerical_constants import EPSILON
from core.statistical.statistical_validation import StationarityValidator, StationarityTestResult

from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)
from iot_machine_learning.infrastructure.ml.cognitive.hyperparameters import HyperparameterAdaptor

from .smoothing import (
    ema,
    holt_stable,
    compute_residual_std,
    compute_confidence,
    classify_trend,
)
from .stationarity import StationarityHandler
from .hyperparam_loader import HyperparameterLoader
from .optimizer import ParameterOptimizer

logger = logging.getLogger(__name__)


class StatisticalPredictionEngine(PredictionEngine):
    """EMA/Holt-based prediction engine with adaptive parameters."""
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        horizon: int = 1,
        series_id: Optional[str] = None,
        enable_optimization: bool = True,
        hyperparameter_adaptor: Optional[HyperparameterAdaptor] = None,
        max_trend_ratio: float = 0.5,
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
        self._horizon = horizon
        self._max_trend_ratio = max_trend_ratio
        
        # Learning state
        self._prediction_history: Deque[float] = deque(maxlen=100)
        self._prediction_count = 0
        self._needs_reoptimization = False
        self._current_mae = 999.0
        
        # Composed components
        self._stationarity = StationarityHandler(stationarity_validator)
        self._hyperparam_loader = HyperparameterLoader(
            hyperparameter_adaptor, series_id, "statistical_ema_holt"
        )
        self._optimizer = ParameterOptimizer(
            hyperparameter_adaptor, series_id, "statistical_ema_holt"
        )
        
        # Current parameters (will be overwritten by _load_hyperparams)
        self._alpha = alpha
        self._beta = beta
    
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
        """Predict next value using EMA or Holt's method."""
        if not values:
            raise ValueError("values cannot be empty")
        
        # Load hyperparameters before prediction
        self._alpha, self._beta, self._current_mae = self._hyperparam_loader.load(
            self._alpha, self._beta, self._current_mae, len(values)
        )
        
        n = len(values)
        if not self.can_handle(n):
            return self._fallback(values)
        
        # Validate stationarity
        is_stationary = self._stationarity.validate(values)
        
        if not is_stationary:
            return self._stationarity.create_ema_fallback(values, self._alpha, self._horizon)
        
        # Holt's double exponential smoothing with stability
        level, trend = holt_stable(values, self._alpha, self._beta, self._max_trend_ratio)
        predicted = level + trend * self._horizon
        
        # Residual analysis
        ema_series = ema(values, self._alpha)
        residual_std = compute_residual_std(values, ema_series)
        confidence = compute_confidence(values, residual_std)
        trend_dir = classify_trend(trend, residual_std)
        
        metadata = {
            "level": round(level, 6),
            "trend_component": round(trend, 6),
            "alpha": self._alpha,
            "beta": self._beta,
            "residual_std": round(residual_std, 6),
            "horizon_steps": self._horizon,
            "fallback": None,
            "diagnostic": {
                "stability_indicator": round(min(1.0, residual_std / (abs(sum(values) / n) + EPSILON.DIVISION) if n > 0 else 1.0), 4),
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
        """Record actual value for online alpha adjustment + re-optimization."""
        if not self._enable_optimization:
            return
        
        error = abs(predicted - actual)
        self._prediction_history.append(actual)
        self._prediction_count += 1
        
        # Online alpha micro-adjustment
        self._online_adjust_alpha(error, actual)
        
        # Deferred full re-optimization threshold
        if self._prediction_count >= 20 and len(self._prediction_history) >= 20:
            self._needs_reoptimization = True
    
    def _online_adjust_alpha(self, error: float, actual: float) -> None:
        """Micro-adjust alpha based on recent prediction error."""
        scale = max(abs(actual), 1.0)
        normalized_error = error / scale
        lr = 0.01
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
        """Trigger deferred re-optimization if threshold was reached."""
        if not self._needs_reoptimization:
            return
        
        new_alpha, new_beta, new_mae, improved = self._optimizer.optimize(
            self._prediction_history,
            self._alpha,
            self._beta,
            self._current_mae,
        )
        
        if improved:
            self._alpha = new_alpha
            self._beta = new_beta
            self._current_mae = new_mae
            self._hyperparam_loader.save(new_alpha, new_beta, new_mae)
        
        self._needs_reoptimization = False
        self._prediction_count = 0
    
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