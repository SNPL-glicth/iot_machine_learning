"""Temporal Z-score sub-detectors — velocidad y aceleración.

Cada clase tiene UNA responsabilidad: evaluar si la velocidad o
aceleración del último punto es anómala respecto a la distribución
histórica.

Sin sklearn, sin I/O.
"""

from __future__ import annotations

from typing import List, Optional

from ..detector_protocol import SubDetector
from ..scoring_functions import compute_z_score, compute_z_vote
from ..temporal_stats import TemporalTrainingStats, compute_temporal_training_stats


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
        return compute_z_vote(z, self._lower, self._upper)

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
        return compute_z_vote(z, self._lower, self._upper)

    @property
    def is_trained(self) -> bool:
        return self._temporal_stats.has_temporal
