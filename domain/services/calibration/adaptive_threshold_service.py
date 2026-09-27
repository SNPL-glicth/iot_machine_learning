"""Adaptive Threshold Calibration Domain Service.

Conforms to:
- ISO/IEC 25010:2023: Performance efficiency, reliability, functional suitability.
- ISO/IEC 22989:2022: Continuous learning, auditability, explainability.
- Strict constraint: Line count <= 180 lines, pure domain logic.
"""

from __future__ import annotations

import math
import time
from typing import List

from domain.entities.calibration.threshold_state import (
    AdaptiveThresholdState,
    DynamicConfidenceBand,
    ThresholdAuditSnapshot,
)


class AdaptiveThresholdService:
    """Computes streaming statistical thresholds adapting to local volatility."""

    def __init__(
        self,
        decay_factor: float = 0.98,
        default_multiplier_k: float = 2.5,
        default_floor: float = 1.5,
        default_ceiling: float = 6.0,
    ) -> None:
        """Initialize service with memory decay and safety bounds."""
        self._decay = max(0.80, min(0.9999, float(decay_factor)))
        self._k = max(1.0, float(default_multiplier_k))
        self._floor = max(0.1, float(default_floor))
        self._ceiling = max(self._floor + 0.5, float(default_ceiling))

    def initialize_state(
        self,
        initial_mean: float = 2.0,
        initial_variance: float = 0.25,
        multiplier_k: float | None = None,
        floor_threshold: float | None = None,
        ceiling_threshold: float | None = None,
        timestamp_ns: int | None = None,
    ) -> AdaptiveThresholdState:
        """Construct initial baseline threshold state."""
        k_val = self._k if multiplier_k is None else max(1.0, float(multiplier_k))
        f_val = self._floor if floor_threshold is None else max(0.1, float(floor_threshold))
        c_val = self._ceiling if ceiling_threshold is None else max(f_val + 0.5, float(ceiling_threshold))
        var_val = max(1e-6, float(initial_variance))
        mean_val = float(initial_mean)

        std_val = math.sqrt(var_val)
        raw_th = mean_val + (k_val * std_val)
        clamped_th = max(f_val, min(c_val, raw_th))
        t_ns = time.time_ns() if timestamp_ns is None else timestamp_ns

        return AdaptiveThresholdState(
            sample_count=1,
            running_mean=mean_val,
            running_variance=var_val,
            current_threshold=clamped_th,
            floor_threshold=f_val,
            ceiling_threshold=c_val,
            multiplier_k=k_val,
            last_update_timestamp_ns=t_ns,
        )

    def update_threshold(
        self,
        observed_metric: float,
        state: AdaptiveThresholdState,
        timestamp_ns: int | None = None,
    ) -> AdaptiveThresholdState:
        """Update running moments online via exponential forgetting Welford."""
        val = max(0.0, float(observed_metric))
        alpha = 1.0 - self._decay

        delta = val - state.running_mean
        new_mean = state.running_mean + (alpha * delta)
        # Update variance with exponential memory weighting
        new_var = (self._decay * state.running_variance) + (alpha * (delta ** 2))
        new_var = max(1e-6, new_var)

        std_val = math.sqrt(new_var)
        raw_threshold = new_mean + (state.multiplier_k * std_val)
        clamped_threshold = max(state.floor_threshold, min(state.ceiling_threshold, raw_threshold))

        t_ns = time.time_ns() if timestamp_ns is None else timestamp_ns

        return AdaptiveThresholdState(
            sample_count=state.sample_count + 1,
            running_mean=new_mean,
            running_variance=new_var,
            current_threshold=clamped_threshold,
            floor_threshold=state.floor_threshold,
            ceiling_threshold=state.ceiling_threshold,
            multiplier_k=state.multiplier_k,
            last_update_timestamp_ns=t_ns,
        )

    def evaluate_admissibility(
        self,
        observed_value: float,
        state: AdaptiveThresholdState,
        metric_name: str = "mahalanobis",
        timestamp_ns: int | None = None,
    ) -> ThresholdAuditSnapshot:
        """Evaluate observation against calibrated adaptive threshold."""
        obs = float(observed_value)
        admissible = obs <= state.current_threshold
        t_ns = time.time_ns() if timestamp_ns is None else timestamp_ns

        return ThresholdAuditSnapshot(
            metric_name=metric_name,
            observed_value=obs,
            threshold_applied=state.current_threshold,
            is_admissible=admissible,
            state=state,
            timestamp_ns=t_ns,
        )

    def compute_adaptive_confidence_bands(
        self,
        base_dispersion: float,
    ) -> List[DynamicConfidenceBand]:
        """Compute volatility-adaptive confidence bands for MoE gating."""
        sigma = max(0.001, min(0.05, float(base_dispersion)))

        return [
            DynamicConfidenceBand(
                min_confidence=0.9, max_confidence=1.0, action_scale=1.0,
                stop_ratio=float(sigma * 1.5), target_ratio=float(sigma * 2.5), horizon_steps=20,
            ),
            DynamicConfidenceBand(
                min_confidence=0.7, max_confidence=0.9, action_scale=0.7,
                stop_ratio=float(sigma * 1.2), target_ratio=float(sigma * 1.8), horizon_steps=15,
            ),
            DynamicConfidenceBand(
                min_confidence=0.5, max_confidence=0.7, action_scale=0.4,
                stop_ratio=float(sigma * 1.0), target_ratio=float(sigma * 1.4), horizon_steps=12,
            ),
            DynamicConfidenceBand(
                min_confidence=0.3, max_confidence=0.5, action_scale=0.2,
                stop_ratio=float(sigma * 0.8), target_ratio=float(sigma * 1.0), horizon_steps=10,
            ),
            DynamicConfidenceBand(
                min_confidence=0.0, max_confidence=0.3, action_scale=0.0,
                stop_ratio=0.0, target_ratio=0.0, horizon_steps=0,
            ),
        ]
