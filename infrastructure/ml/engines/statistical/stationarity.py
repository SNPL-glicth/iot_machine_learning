"""Stationarity validation and fallback handling."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from core.statistical.statistical_validation import StationarityValidator, StationarityTestResult
from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
from .smoothing import ema, compute_residual_std, compute_confidence

logger = logging.getLogger(__name__)


class StationarityHandler:
    """Handles stationarity validation and EMA fallback."""
    
    def __init__(self, validator: Optional[StationarityValidator] = None):
        self._validator = validator
        self._result: Optional[StationarityTestResult] = None
    
    @property
    def result(self) -> Optional[StationarityTestResult]:
        return self._result
    
    def validate(self, values: List[float]) -> bool:
        """Validate stationarity. Returns True if stationary, False otherwise."""
        if self._validator is None or len(values) < self._validator.min_samples:
            return True
        
        data_array = np.array(values)
        self._result = self._validator.validate(data_array)
        
        logger.info(
            "statistical_stationarity_validation",
            extra={
                "is_stationary": self._result.is_stationary,
                "stationarity_type": self._result.stationarity_type.value,
                "recommendation": self._result.recommendation,
                "adf_p": self._result.adf_p_value,
            },
        )
        
        return self._result.is_stationary
    
    def create_ema_fallback(
        self,
        values: List[float],
        alpha: float,
        horizon: int,
    ) -> PredictionResult:
        """Create EMA fallback prediction for non-stationary data."""
        ema_series = ema(values, alpha)
        predicted = ema_series[-1]
        residual_std = compute_residual_std(values, ema_series)
        
        n = len(values)
        mean_abs = abs(sum(values) / n) if n > 0 else 1.0
        noise_ratio = residual_std / (mean_abs + 1e-12)  # EPSILON.DIVISION
        confidence = max(0.2, min(0.95, 1.0 - noise_ratio))
        trend_dir = "stable"
        stability = min(1.0, noise_ratio)
        
        metadata = {
            "level": predicted,
            "trend_component": 0.0,
            "alpha": alpha,
            "beta": 0.0,
            "residual_std": round(residual_std, 6),
            "horizon_steps": horizon,
            "fallback": "ema_instead_of_holt",
            "diagnostic": {
                "stability_indicator": round(stability, 4),
                "local_fit_error": round(residual_std, 6),
                "method": "ema_only",
            },
        }
        
        logger.warning(
            "statistical_non_stationary_fallback",
            extra={"recommendation": "use_ema"},
        )
        
        return PredictionResult(
            predicted_value=predicted,
            confidence=confidence,
            trend=trend_dir,
            metadata=metadata,
        )