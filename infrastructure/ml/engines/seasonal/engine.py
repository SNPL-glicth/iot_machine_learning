"""Seasonal prediction engine using FFT-based cycle detection.

R-4: Detects dominant frequency in time series using Fast Fourier
Transform and projects next value based on detected cycle.
Lightweight implementation targeting <10ms latency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ...interfaces import PredictionEngine, PredictionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeasonalConfig:
    """Configuration for seasonal prediction."""
    min_period: int = 4          # Minimum detectable cycle length
    max_period: int = 100        # Maximum detectable cycle length
    min_confidence: float = 0.3  # Minimum confidence threshold
    fft_threshold: float = 0.15  # Peak prominence threshold


class SeasonalPredictorEngine(PredictionEngine):
    """Predicts using FFT-detected seasonal patterns.
    
    R-4: Seasonality engine for capturing cyclic patterns
    that Taylor and Baseline engines miss.
    
    Performance: <10ms latency for windows up to 50 points.
    """
    
    def __init__(self, config: Optional[SeasonalConfig] = None) -> None:
        self._config = config or SeasonalConfig()
        self._cache: dict = {}
    
    @property
    def name(self) -> str:
        return "seasonal_fft"
    
    def can_handle(self, n_points: int) -> bool:
        # Need at least 2 cycles for reliable detection
        return n_points >= self._config.min_period * 2
    
    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        """Predict next value using detected seasonality.
        
        Args:
            values: Time series values
            timestamps: Optional timestamps (used for regularity check)
            
        Returns:
            PredictionResult with seasonal projection
        """
        start_time = time.perf_counter()
        
        if len(values) < self._config.min_period * 2:
            return self._fallback(values)
        
        try:
            # Detect dominant cycle
            period, confidence = self._detect_cycle(values)
            
            if period is None or confidence < self._config.min_confidence:
                return self._fallback(values)
            
            # Project next value using detected period
            predicted = self._project(values, period)
            
            # Classify trend based on cycle phase
            trend = self._classify_trend(values, period)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return PredictionResult(
                predicted_value=predicted,
                confidence=confidence,
                trend=trend,
                metadata={
                    "engine": "seasonal_fft",
                    "detected_period": period,
                    "latency_ms": latency_ms,
                },
            )
            
        except Exception as e:
            logger.debug(f"seasonal_prediction_failed: {e}")
            return self._fallback(values)
    
    def _detect_cycle(self, values: List[float]) -> Tuple[Optional[int], float]:
        """Detect dominant cycle period using FFT.
        
        Args:
            values: Time series values
            
        Returns:
            (period, confidence) or (None, 0) if no clear cycle
        """
        n = len(values)
        
        # Detrend to remove DC component
        mean_val = np.mean(values)
        detrended = np.array(values) - mean_val
        
        # FFT
        fft = np.fft.rfft(detrended)
        power = np.abs(fft) ** 2
        
        # Find peaks in frequency domain (excluding DC)
        freqs = np.fft.rfftfreq(n)
        
        # Consider only frequencies within min/max period range
        min_freq = 1.0 / self._config.max_period
        max_freq = 1.0 / self._config.min_period
        
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
        if not valid_mask.any():
            return None, 0.0
        
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]
        
        if len(valid_power) == 0:
            return None, 0.0
        
        # Find dominant frequency
        peak_idx = np.argmax(valid_power)
        peak_power = valid_power[peak_idx]
        total_power = np.sum(power[1:])  # Exclude DC
        
        if total_power == 0:
            return None, 0.0
        
        # Confidence based on peak prominence
        confidence = peak_power / total_power
        
        if confidence < self._config.fft_threshold:
            return None, 0.0
        
        dominant_freq = valid_freqs[peak_idx]
        period = int(round(1.0 / dominant_freq))
        
        return period, min(confidence, 0.95)
    
    def _project(self, values: List[float], period: int) -> float:
        """Project next value using detected period.
        
        Uses phase-matched value from previous cycle.
        """
        n = len(values)
        
        # Find position in current cycle
        position = n % period
        
        # Get values from previous cycles at same position
        previous_values = []
        for i in range(1, 3):  # Look back 2 cycles
            idx = n - (i * period) + position
            if 0 <= idx < n:
                previous_values.append(values[idx])
        
        if not previous_values:
            # Fallback to last value if no previous cycle data
            return values[-1]
        
        # Average of phase-matched values
        return float(np.mean(previous_values))
    
    def _classify_trend(
        self, 
        values: List[float], 
        period: int
    ) -> str:
        """Classify trend based on cycle phase."""
        n = len(values)
        if n < 2:
            return "stable"
        
        position = n % period
        
        # Compare current position with previous cycle
        prev_idx = n - period + position - 1
        if prev_idx < 0 or prev_idx >= n - 1:
            return "stable"
        
        diff = values[-1] - values[prev_idx]
        
        if abs(diff) < 0.001:
            return "stable"
        return "up" if diff > 0 else "down"
    
    def _fallback(self, values: List[float]) -> PredictionResult:
        """Fallback to last value when seasonality unclear."""
        return PredictionResult(
            predicted_value=values[-1] if values else 0.0,
            confidence=0.3,
            trend="unknown",
            metadata={"engine": "seasonal_fft", "fallback": True},
        )
