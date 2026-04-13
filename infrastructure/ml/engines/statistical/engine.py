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

from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
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
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        
        self._series_id = series_id
        self._enable_optimization = enable_optimization
        self._prediction_history: Deque[float] = deque(maxlen=100)
        self._prediction_count = 0
        self._current_mae = 999.0
        
        # Try to load optimized params from DB
        if series_id and enable_optimization:
            loaded = self._load_params(series_id)
            if loaded:
                self._alpha, self._beta, self._current_mae = loaded
                logger.info(
                    "statistical_params_loaded_from_db",
                    extra={
                        "series_id": series_id,
                        "alpha": self._alpha,
                        "beta": self._beta,
                        "mae": round(self._current_mae, 4),
                    },
                )
            else:
                self._alpha = alpha
                self._beta = beta
        else:
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
    
    def record_actual(self, predicted: float, actual: float) -> None:
        """Record actual value for re-optimization.
        
        Args:
            predicted: Predicted value
            actual: Actual observed value
        """
        if not self._enable_optimization:
            return
        
        self._prediction_history.append(actual)
        self._prediction_count += 1
        
        # Re-optimize after 50 predictions
        if self._prediction_count >= 50 and len(self._prediction_history) >= 20:
            self._reoptimize()
            self._prediction_count = 0
    
    def _load_params(self, series_id: str) -> Optional[tuple[float, float, float]]:
        """Load parameters from DB.
        
        Args:
            series_id: Series identifier
            
        Returns:
            Tuple of (alpha, beta, mae) or None
        """
        try:
            from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.statistical_params_repository import (
                StatisticalParamsRepository,
            )
            repo = StatisticalParamsRepository()
            return repo.load_params(series_id)
        except Exception as exc:
            logger.warning(
                "statistical_params_load_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )
            return None
    
    def _reoptimize(self) -> None:
        """Re-optimize parameters if MAE improves > 5%."""
        if not self._series_id:
            return
        
        try:
            from iot_machine_learning.infrastructure.ml.engines.statistical.param_optimizer import (
                StatisticalParamOptimizer,
            )
            from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.statistical_params_repository import (
                StatisticalParamsRepository,
            )
            
            optimizer = StatisticalParamOptimizer()
            values = list(self._prediction_history)
            
            new_alpha, new_beta, new_mae = optimizer.optimize(values)
            
            # Only update if MAE improves > 5%
            improvement = (self._current_mae - new_mae) / self._current_mae
            if improvement > 0.05:
                self._alpha = new_alpha
                self._beta = new_beta
                self._current_mae = new_mae
                
                # Persist to DB
                repo = StatisticalParamsRepository()
                repo.save_params(
                    series_id=self._series_id,
                    alpha=new_alpha,
                    beta=new_beta,
                    mae=new_mae,
                    n_samples=len(values),
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
