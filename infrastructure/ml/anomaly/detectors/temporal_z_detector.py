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

    def train(self, values: List[float], **kwargs: Any) -> None:
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            return
        ts_list = list(timestamps)
        self._temporal_stats = compute_temporal_training_stats(
            values, ts_list
        )
        try:
            from iot_machine_learning.domain.validators.temporal_features import (
                compute_temporal_features,
            )
            tf = compute_temporal_features(values, ts_list)
            if tf.has_velocity and self._temporal_stats.has_temporal:
                self._training_raw_scores = [
                    float(compute_z_score(v, self._temporal_stats.vel_mean, self._temporal_stats.vel_std))
                    for v in tf.velocities
                ]
            else:
                self._training_raw_scores = []
        except Exception:
            self._training_raw_scores = []

    def vote(self, value: float, **kwargs: Any) -> Optional[float]:
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
        
        raw_s = self.raw_score(value, **kwargs)
        if raw_s is None:
            return None
        return compute_z_vote(raw_s, lower, upper)

    def raw_score(self, value: float, **kwargs: Any) -> Optional[float]:
        """Calcula el Z-score continuo de velocidad sin discretizar."""
        if not self._temporal_stats.has_temporal:
            return None

        dynamic_features: Optional[Dict[str, Any]] = kwargs.get("dynamic_features")
        if dynamic_features is not None:
            derivative = dynamic_features.get("derivative")
            if derivative is not None:
                return float(
                    compute_z_score(
                        derivative,
                        self._temporal_stats.vel_mean,
                        self._temporal_stats.vel_std,
                    )
                )

        temporal_features = kwargs.get("temporal_features")
        if temporal_features is not None and getattr(temporal_features, "has_velocity", False):
            return float(
                compute_z_score(
                    temporal_features.last_velocity,
                    self._temporal_stats.vel_mean,
                    self._temporal_stats.vel_std,
                )
            )

        window = kwargs.get("window")
        if window is not None and hasattr(window, "temporal_features"):
            try:
                tf = window.temporal_features
                if getattr(tf, "has_velocity", False):
                    return float(
                        compute_z_score(
                            tf.last_velocity,
                            self._temporal_stats.vel_mean,
                            self._temporal_stats.vel_std,
                        )
                    )
            except Exception:
                pass

        return None

    def get_training_raw_scores(
        self, values: List[float], **kwargs: Any
    ) -> List[float]:
        """Retorna las puntuaciones continuas de entrenamiento para calibración."""
        if hasattr(self, "_training_raw_scores") and self._training_raw_scores:
            return list(self._training_raw_scores)
        if not self._temporal_stats.has_temporal:
            return []
        try:
            from iot_machine_learning.domain.validators.temporal_features import (
                compute_temporal_features,
            )
            timestamps = kwargs.get("timestamps")
            ts_list = list(timestamps) if timestamps is not None else [float(i) for i in range(len(values))]
            tf = compute_temporal_features(values, ts_list)
            if tf.has_velocity:
                return [
                    float(compute_z_score(v, self._temporal_stats.vel_mean, self._temporal_stats.vel_std))
                    for v in tf.velocities
                ]
        except Exception:
            pass
        return []

    @property
    def is_trained(self) -> bool:
        return self._temporal_stats.has_temporal


class AccelerationZDetector(SubDetector):
    """Sub-detector de Z-score de aceleración (d²v/dt²)."""

    def __init__(self, lower: float = 2.0, upper: float = 3.0) -> None:
        self._lower = lower
        self._upper = upper
        self._temporal_stats: TemporalTrainingStats = TemporalTrainingStats.empty()
        self._training_raw_scores: List[float] = []

    @property
    def method_name(self) -> str:
        return "acceleration_z"

    def train(self, values: List[float], **kwargs: Any) -> None:
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            return
        ts_list = list(timestamps)
        self._temporal_stats = compute_temporal_training_stats(
            values, ts_list
        )
        try:
            from iot_machine_learning.domain.validators.temporal_features import (
                compute_temporal_features,
            )
            tf = compute_temporal_features(values, ts_list)
            if tf.has_acceleration and self._temporal_stats.has_temporal:
                self._training_raw_scores = [
                    float(compute_z_score(a, self._temporal_stats.acc_mean, self._temporal_stats.acc_std))
                    for a in tf.accelerations
                ]
            else:
                self._training_raw_scores = []
        except Exception:
            self._training_raw_scores = []

    def vote(self, value: float, **kwargs: Any) -> Optional[float]:
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
        
        raw_s = self.raw_score(value, **kwargs)
        if raw_s is None:
            return None
        return compute_z_vote(raw_s, lower, upper)

    def raw_score(self, value: float, **kwargs: Any) -> Optional[float]:
        """Calcula el Z-score continuo de aceleración sin discretizar."""
        if not self._temporal_stats.has_temporal:
            return None

        dynamic_features: Optional[Dict[str, Any]] = kwargs.get("dynamic_features")
        if dynamic_features is not None:
            second_derivative = dynamic_features.get("second_derivative")
            if second_derivative is not None:
                return float(
                    compute_z_score(
                        second_derivative,
                        self._temporal_stats.acc_mean,
                        self._temporal_stats.acc_std,
                    )
                )

        temporal_features = kwargs.get("temporal_features")
        if temporal_features is not None and getattr(temporal_features, "has_acceleration", False):
            return float(
                compute_z_score(
                    temporal_features.last_acceleration,
                    self._temporal_stats.acc_mean,
                    self._temporal_stats.acc_std,
                )
            )

        window = kwargs.get("window")
        if window is not None and hasattr(window, "temporal_features"):
            try:
                tf = window.temporal_features
                if getattr(tf, "has_acceleration", False):
                    return float(
                        compute_z_score(
                            tf.last_acceleration,
                            self._temporal_stats.acc_mean,
                            self._temporal_stats.acc_std,
                        )
                    )
            except Exception:
                pass

        return None

    def get_training_raw_scores(
        self, values: List[float], **kwargs: Any
    ) -> List[float]:
        """Retorna las puntuaciones continuas de entrenamiento para calibración."""
        if hasattr(self, "_training_raw_scores") and self._training_raw_scores:
            return list(self._training_raw_scores)
        if not self._temporal_stats.has_temporal:
            return []
        try:
            from iot_machine_learning.domain.validators.temporal_features import (
                compute_temporal_features,
            )
            timestamps = kwargs.get("timestamps")
            ts_list = list(timestamps) if timestamps is not None else [float(i) for i in range(len(values))]
            tf = compute_temporal_features(values, ts_list)
            if tf.has_acceleration:
                return [
                    float(compute_z_score(a, self._temporal_stats.acc_mean, self._temporal_stats.acc_std))
                    for a in tf.accelerations
                ]
        except Exception:
            pass
        return []

    @property
    def is_trained(self) -> bool:
        return self._temporal_stats.has_temporal
