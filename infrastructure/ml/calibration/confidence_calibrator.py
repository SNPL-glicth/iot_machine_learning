"""Confidence calibrator — temperature scaling for anomaly scores.

Single responsibility: convert raw anomaly scores into calibrated probabilities
using temperature-scaled sigmoid transformation.

Supports regime-aware temperature adjustment for adaptive calibration.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Calibrates raw anomaly scores into probabilities using temperature scaling.
    
    Uses sigmoid transformation with configurable temperature parameter:
        calibrated = 1 / (1 + exp(-score / temperature))
    
    Higher temperature → smoother probabilities (less confident)
    Lower temperature → sharper probabilities (more confident)
    
    Supports regime-aware temperature adjustment:
    - VOLATILE regime → higher temperature (more conservative)
    - STABLE regime → lower temperature (more confident)
    
    Attributes:
        _base_temperature: Default temperature for calibration.
        _regime_temperatures: Per-regime temperature overrides.
    """
    
    def __init__(
        self,
        temperature: float = 1.5,
        regime_temperatures: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize confidence calibrator.
        
        Args:
            temperature: Base temperature for sigmoid scaling.
            regime_temperatures: Optional per-regime temperature overrides.
                Example: {"VOLATILE": 2.0, "STABLE": 1.2}
        
        Raises:
            ValueError: If temperature <= 0.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        self._base_temperature = temperature
        self._regime_temperatures = regime_temperatures or {}
    
    def calibrate(
        self,
        score: float,
        regime: Optional[str] = None,
    ) -> float:
        """Calibrate raw score into probability using temperature scaling.
        
        Args:
            score: Raw anomaly score [0, +inf).
            regime: Optional regime for temperature adjustment.
        
        Returns:
            Calibrated confidence [0.0, 1.0].
        """
        # Handle edge cases
        if not math.isfinite(score):
            logger.warning(
                "confidence_calibration_invalid_score",
                extra={
                    "event": "WARNING",
                    "score": score,
                    "action_taken": "return_zero",
                },
            )
            return 0.0
        
        # Select temperature based on regime
        temperature = self._get_temperature(regime)
        
        # Apply temperature-scaled sigmoid
        try:
            # sigmoid(x) = 1 / (1 + exp(-x))
            # temperature scaling: x' = x / T
            scaled_score = score / temperature
            
            # Numerical stability: clip to avoid overflow
            # exp(-x) overflows when x < -700
            # exp(x) overflows when x > 700
            scaled_score = max(-700.0, min(700.0, scaled_score))
            
            calibrated = 1.0 / (1.0 + math.exp(-scaled_score))
            
            logger.debug(
                "confidence_calibration",
                extra={
                    "event": "CALIBRATION",
                    "raw_score": round(score, 4),
                    "calibrated_confidence": round(calibrated, 4),
                    "temperature": temperature,
                    "regime": regime,
                },
            )
            
            return calibrated
        
        except Exception as e:
            logger.error(
                "confidence_calibration_failed",
                extra={
                    "event": "CALIBRATION_ERROR",
                    "error": str(e),
                    "score": score,
                    "temperature": temperature,
                    "action_taken": "return_zero",
                },
            )
            return 0.0
    
    def _get_temperature(self, regime: Optional[str]) -> float:
        """Get temperature for given regime.
        
        Args:
            regime: Regime name (e.g., "VOLATILE", "STABLE").
        
        Returns:
            Temperature value.
        """
        if regime and regime in self._regime_temperatures:
            return self._regime_temperatures[regime]
        return self._base_temperature
    
    @property
    def base_temperature(self) -> float:
        """Base temperature value."""
        return self._base_temperature
    
    @property
    def regime_temperatures(self) -> Dict[str, float]:
        """Per-regime temperature overrides."""
        return self._regime_temperatures.copy()
