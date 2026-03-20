"""Vote context builder for VotingAnomalyDetector.

Pure functions that build the kwargs dict passed to each sub-detector's
vote() call, and extract temporal z-scores for the narrator.

No state, no I/O — extracted from VotingAnomalyDetector to keep that
class focused on orchestration only.
"""

from __future__ import annotations

from typing import Optional

from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from ..scoring import compute_z_score, TemporalTrainingStats


def build_vote_context(
    window: SensorWindow,
    temporal_stats: TemporalTrainingStats,
) -> dict:
    """Build kwargs context for sub-detector vote() calls.

    Args:
        window: Current sensor window.
        temporal_stats: Temporal training statistics (may be empty).

    Returns:
        Dict with optional keys: ``temporal_features``, ``nd_features``.
    """
    ctx: dict = {}

    if temporal_stats.has_temporal and window.size >= 2:
        try:
            ctx["temporal_features"] = window.temporal_features
        except Exception:
            pass

    if temporal_stats.has_temporal and window.size >= 3:
        try:
            import numpy as np
            tf = window.temporal_features
            if tf.has_velocity and tf.has_acceleration:
                ctx["nd_features"] = np.array([[
                    window.last_value,
                    tf.last_velocity,
                    tf.last_acceleration,
                ]])
        except Exception:
            pass

    return ctx


def extract_vel_z(
    window: SensorWindow,
    temporal_stats: TemporalTrainingStats,
) -> float:
    """Extract velocity z-score for the anomaly narrator.

    Args:
        window: Current sensor window.
        temporal_stats: Temporal training statistics.

    Returns:
        Velocity z-score, or 0.0 if unavailable.
    """
    if not temporal_stats.has_temporal or window.size < 2:
        return 0.0
    try:
        tf = window.temporal_features
        if tf.has_velocity:
            return compute_z_score(
                tf.last_velocity,
                temporal_stats.vel_mean,
                temporal_stats.vel_std,
            )
    except Exception:
        pass
    return 0.0


def extract_acc_z(
    window: SensorWindow,
    temporal_stats: TemporalTrainingStats,
) -> float:
    """Extract acceleration z-score for the anomaly narrator.

    Args:
        window: Current sensor window.
        temporal_stats: Temporal training statistics.

    Returns:
        Acceleration z-score, or 0.0 if unavailable.
    """
    if not temporal_stats.has_temporal or window.size < 3:
        return 0.0
    try:
        tf = window.temporal_features
        if tf.has_acceleration:
            return compute_z_score(
                tf.last_acceleration,
                temporal_stats.acc_mean,
                temporal_stats.acc_std,
            )
    except Exception:
        pass
    return 0.0
