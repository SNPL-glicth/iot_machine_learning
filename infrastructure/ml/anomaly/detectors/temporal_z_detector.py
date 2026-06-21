"""Temporal Z-score sub-detectors — velocidad y aceleración.

Cada clase tiene UNA responsabilidad: evaluar si la velocidad o
aceleración del último punto es anómala respecto a la distribución
histórica.

Sin sklearn, sin I/O.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

from ..core.protocol import SubDetector
from ..scoring.functions import compute_z_score, compute_z_vote
from ..scoring.temporal import TemporalTrainingStats, compute_temporal_training_stats


class VelocityZDetector(SubDetector):
    """Sub-detector de Z-score de velocidad (dv/dt)."""

    def __init__(self, lower: float = 2.0, upper: float = 3.0) -> None:
        self._lower = lower
        self._upper = upper
        self._temporal_stats: TemporalTrainingStats = TemporalTrainingStats.empty()

    @property
    def method_name(self) -> str:
        return "velocity_z"

    def train(self, values: List[float], **kwargs: object) -> None:
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            return
        self._temporal_stats = compute_temporal_training_stats(
            values, list(timestamps)
        )

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if not self._temporal_stats.has_temporal:
            return None
        
        # Get regime for contextual thresholds (ETAPA 2)
        regime = kwargs.get("regime")
        regime_config = kwargs.get("regime_config")
        
        # Adjust thresholds according to regime (from config if available)
        lower = self._lower
        upper = self._upper
        if regime and regime_config:
            if regime == "STARTUP":
                lower = regime_config.velocity_z_lower_startup
                upper = regime_config.velocity_z_upper_startup
            elif regime == "SHUTDOWN":
                lower = regime_config.velocity_z_lower_shutdown
                upper = regime_config.velocity_z_upper_shutdown
            elif regime == "STABLE_NORMAL":
                lower = regime_config.velocity_z_lower_stable
                upper = regime_config.velocity_z_upper_stable
        
        # Try to use DynamicFeatures derivative first (v2.0.0)
        dynamic_features: Optional[Dict[str, Any]] = kwargs.get("dynamic_features")
        if dynamic_features is not None:
            derivative = dynamic_features.get("derivative")
            if derivative is not None:
                z = compute_z_score(
                    derivative,
                    self._temporal_stats.vel_mean,
                    self._temporal_stats.vel_std,
                )
                return compute_z_vote(z, lower, upper)
        
        # Fallback to temporal_features (v1.0.0)
        temporal_features = kwargs.get("temporal_features")
        if temporal_features is None:
            return None
        if not temporal_features.has_velocity:
            return None
        z = compute_z_score(
            temporal_features.last_velocity,
            self._temporal_stats.vel_mean,
            self._temporal_stats.vel_std,
        )
        return compute_z_vote(z, lower, upper)

    @property
    def is_trained(self) -> bool:
        return self._temporal_stats.has_temporal


class AccelerationZDetector(SubDetector):
    """Sub-detector de Z-score de aceleración (d²v/dt²)."""

    def __init__(self, lower: float = 2.0, upper: float = 3.0) -> None:
        self._lower = lower
        self._upper = upper
        self._temporal_stats: TemporalTrainingStats = TemporalTrainingStats.empty()

    @property
    def method_name(self) -> str:
        return "acceleration_z"

    def train(self, values: List[float], **kwargs: object) -> None:
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            return
        self._temporal_stats = compute_temporal_training_stats(
            values, list(timestamps)
        )

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if not self._temporal_stats.has_temporal:
            return None
        
        # Get regime for contextual thresholds (ETAPA 2)
        regime = kwargs.get("regime")
        regime_config = kwargs.get("regime_config")
        
        # Adjust thresholds according to regime (from config if available)
        lower = self._lower
        upper = self._upper
        if regime and regime_config:
            if regime == "STARTUP":
                lower = regime_config.acceleration_z_lower_startup
                upper = regime_config.acceleration_z_upper_startup
            elif regime == "SHUTDOWN":
                lower = regime_config.acceleration_z_lower_shutdown
                upper = regime_config.acceleration_z_upper_shutdown
            elif regime == "STABLE_NORMAL":
                lower = regime_config.acceleration_z_lower_stable
                upper = regime_config.acceleration_z_upper_stable
        
        # Try to use DynamicFeatures second_derivative first (v2.0.0)
        dynamic_features: Optional[Dict[str, Any]] = kwargs.get("dynamic_features")
        if dynamic_features is not None:
            second_derivative = dynamic_features.get("second_derivative")
            if second_derivative is not None:
                z = compute_z_score(
                    second_derivative,
                    self._temporal_stats.acc_mean,
                    self._temporal_stats.acc_std,
                )
                return compute_z_vote(z, lower, upper)
        
        # Fallback to temporal_features (v1.0.0)
        temporal_features = kwargs.get("temporal_features")
        if temporal_features is None:
            return None
        if not temporal_features.has_acceleration:
            return None
        z = compute_z_score(
            temporal_features.last_acceleration,
            self._temporal_stats.acc_mean,
            self._temporal_stats.acc_std,
        )
        return compute_z_vote(z, lower, upper)

    @property
    def is_trained(self) -> bool:
        return self._temporal_stats.has_temporal
