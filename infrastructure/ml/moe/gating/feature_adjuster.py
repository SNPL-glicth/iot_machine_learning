"""Feature-based weight adjustments for ContextualRegimeGating."""

from __future__ import annotations

from typing import Dict

from ..feature_context import FeatureContext
from .sigmoid_boost import sigmoid_boost, sigmoid_reduce


class FeatureAdjuster:
    """Applies bounded sigmoid adjustments to expert weights based on features."""
    
    def __init__(
        self,
        slope_threshold: float = 0.01,
    ):
        self._slope_threshold = slope_threshold
    
    def adjust_by_std(
        self, weights: Dict[str, float], std: float
    ) -> Dict[str, float]:
        """Bounded boost [1x, 3x) to high-volatility experts via sigmoid."""
        if std < 0.5:
            return weights
        boost = sigmoid_boost(std, midpoint=1.5, steepness=1.5)
        adjusted = dict(weights)
        for eid in ("taylor", "kalman"):
            if eid in adjusted:
                adjusted[eid] = adjusted[eid] * boost
        if std > 1.5 and "baseline" in adjusted:
            adjusted["baseline"] *= sigmoid_reduce(std, midpoint=2.0, steepness=1.0)
        return adjusted
    
    def adjust_by_slope(
        self, weights: Dict[str, float], slope: float
    ) -> Dict[str, float]:
        """Bounded boost [1x, 3x) to statistical if trending."""
        if abs(slope) <= self._slope_threshold:
            return weights
        boost = sigmoid_boost(abs(slope), midpoint=0.05, steepness=30.0)
        adjusted = dict(weights)
        if "statistical" in adjusted:
            adjusted["statistical"] = adjusted["statistical"] * boost
        if "taylor" in adjusted:
            adjusted["taylor"] = adjusted["taylor"] * (1.0 + (boost - 1.0) * 0.5)
        return adjusted
    
    def adjust_by_noise(
        self, weights: Dict[str, float], noise_ratio: float
    ) -> Dict[str, float]:
        """Bounded boost [1x, 3x) to kalman if noisy."""
        if noise_ratio < 0.15:
            return weights
        boost = sigmoid_boost(noise_ratio, midpoint=0.3, steepness=5.0)
        adjusted = dict(weights)
        if "kalman" in adjusted:
            adjusted["kalman"] = adjusted["kalman"] * boost
        if noise_ratio > 0.4 and "baseline" in adjusted:
            adjusted["baseline"] *= sigmoid_reduce(noise_ratio, midpoint=0.5, steepness=3.0)
        return adjusted
    
    def adjust_by_curvature(
        self, weights: Dict[str, float], curvature: float
    ) -> Dict[str, float]:
        """Adjustment for curvature (acceleration). High curvature -> Taylor."""
        if abs(curvature) <= 0.001:
            return weights
        boost = sigmoid_boost(abs(curvature), midpoint=0.01, steepness=50.0)
        adjusted = dict(weights)
        if "taylor" in adjusted:
            adjusted["taylor"] = adjusted["taylor"] * boost
        return adjusted