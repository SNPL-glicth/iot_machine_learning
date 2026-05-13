"""Confidence calibrator — temperature scaling for anomaly scores.

Single responsibility: convert raw anomaly scores into calibrated probabilities
using temperature-scaled sigmoid transformation.

Supports regime-aware temperature adjustment for adaptive calibration.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Dict

from core.parameters.numerical_constants import CONFIDENCE

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Calibrates raw anomaly scores into probabilities using temperature scaling.

    Uses sigmoid transformation with configurable temperature parameter:
        calibrated = 1 / (1 + exp(-score / temperature))

    Higher temperature → smoother probabilities (less confident)
    Lower temperature → sharper probabilities (more confident)

    Supports regime-aware temperature adjustment:
    - VOLATILE regime → higher temperature (more conservative)
    - STABLE regime → lower temperature (more confident)

    CUÁNDO USAR ESTE MÉTODO:
    - Anomaly scores: para scores de anomalía en [0, +inf), NO para confidence
    - Post-detection: calibración probabilística DESPUÉS de detección de anomalías
    - Regime-aware: cuando se conoce el régimen para ajuste adaptativo
    - NO usar para: confidence values en [0, 1] (usar core/tuning/temperature_scaling.py)
    - NO usar para: pre-decisión por calidad de datos (usar domain/services/confidence_calibrator.py)

    DIFERENCIA con otros calibradores:
    - Este módulo: sigmoid para anomaly scores [0, +inf)
    - core/tuning/temperature_scaling.py: sigmoid centrado para confidence [0, 1]
    - domain/services/confidence_calibrator.py: penalidades aditivas para calidad de datos

    FÓRMULA DE TEMPERATURA (DIFERENCIA INTENCIONAL):
    - Este módulo: sigmoid(score / T) donde score ∈ [0, +inf)
      * NO centra en 0.5 porque anomaly scores no tienen punto medio natural
      * score=0 → calibrated≈0.5, score→∞ → calibrated→1.0
    - TemperatureScaler: sigmoid((c - 0.5) / T) donde c ∈ [0, 1]
      * Centra en 0.5 porque confidence tiene punto medio natural
      * c=0.5 → calibrated=0.5, c=1.0 → calibrated>0.5

    Attributes:
        _base_temperature: Default temperature for calibration.
        _regime_temperatures: Per-regime temperature overrides.
    """
    
    def __init__(
        self,
        temperature: float = CONFIDENCE.TEMP_DEFAULT,
        regime_temperatures: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize confidence calibrator.
        
        Args:
            temperature: Base temperature for sigmoid scaling. Defaults to CONFIDENCE.TEMP_DEFAULT.
            regime_temperatures: Optional per-regime temperature overrides.
                If None, uses CONFIDENCE.TEMP_* values.
                Example: {"VOLATILE": 2.0, "STABLE": 1.2}
        
        Raises:
            ValueError: If temperature <= 0.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        self._base_temperature = temperature
        # Use CONFIDENCE singleton if no regime_temperatures provided
        self._regime_temperatures = regime_temperatures or {
            "STABLE": CONFIDENCE.TEMP_STABLE,
            "TRENDING": CONFIDENCE.TEMP_TRENDING,
            "VOLATILE": CONFIDENCE.TEMP_VOLATILE,
            "NOISY": CONFIDENCE.TEMP_NOISY,
        }
    
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
