"""MoE Context Builder: Constructs FeatureContext from raw values."""

from __future__ import annotations

import statistics
from typing import List

from ..feature_context import FeatureContext


class MoEContextBuilder:
    """Builds FeatureContext from time series values for MoE gating."""
    
    def build(self, values: List[float]) -> FeatureContext:
        """Construct FeatureContext with multi-factor regime classification."""
        n = len(values)
        mean = sum(values) / n if n > 0 else 0.0
        std = statistics.stdev(values) if n >= 2 else 0.0
        
        # Slope via OLS regression
        slope = 0.0
        r_squared = 0.0
        curvature = 0.0
        if n >= 2:
            x_mean = (n - 1) / 2.0
            y_mean = mean
            num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if abs(den) > 1e-12 else 0.0
            if den > 1e-12:
                ss_res = sum((y - (mean + slope * (i - x_mean))) ** 2 for i, y in enumerate(values))
                ss_tot = sum((y - y_mean) ** 2 for y in values)
                r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        
        # Curvature via second differences
        if n >= 3:
            second_diffs = [values[i] - 2 * values[i - 1] + values[i - 2] for i in range(2, n)]
            curvature = sum(second_diffs) / len(second_diffs)
        
        # Noise ratio (CV with stabilizer)
        noise_ratio = std / (abs(mean) + 1e-6) if abs(mean) > 1e-9 else std / (std + 1e-6)
        
        # Autocorrelation lag-1
        autocorr = 0.0
        if n >= 4:
            lag1 = [(values[i] - mean) * (values[i - 1] - mean) for i in range(1, n)]
            var = sum((v - mean) ** 2 for v in values) + 1e-12
            autocorr = sum(lag1) / var if var > 1e-12 else 0.0
        
        # Multi-factor regime classification
        # Priority: noisy > trending > volatile > stable
        if noise_ratio > 0.5 and std > 0.3:
            regime = "noisy"
        elif r_squared > 0.6 and abs(slope) > 0.005 * (abs(mean) + 1e-6):
            regime = "trending"
        elif std > 0.8 * (abs(mean) + 1e-6) or std > 2.0:
            regime = "volatile"
        elif autocorr > 0.5 and abs(slope) < 0.001:
            regime = "stable"
        else:
            regime = "stable"
        
        # Stability: 1.0 = perfectly stable
        stability = 1.0 / (1.0 + noise_ratio + abs(slope) * 10.0)
        
        return FeatureContext(
            regime=regime,
            mean=mean,
            std=std,
            slope=slope,
            curvature=curvature,
            noise_ratio=noise_ratio,
            stability=stability,
            hampel_outlier_mask=[],
            spatial_correlation_score=0.0,
        )