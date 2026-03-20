"""Scoring and statistics for anomaly detection.

Components:
    - Pure scoring functions (z-score, IQR, weighted voting, consensus)
    - Training statistics (mean, std, quartiles)
    - Temporal statistics (velocity, acceleration distributions)
"""

from .functions import (
    compute_z_score,
    compute_z_vote,
    compute_iqr_bounds,
    compute_iqr_vote,
    weighted_vote,
    compute_consensus_confidence,
)
from .training import TrainingStats, compute_training_stats
from .temporal import TemporalTrainingStats, compute_temporal_training_stats

__all__ = [
    "compute_z_score",
    "compute_z_vote",
    "compute_iqr_bounds",
    "compute_iqr_vote",
    "weighted_vote",
    "compute_consensus_confidence",
    "TrainingStats",
    "compute_training_stats",
    "TemporalTrainingStats",
    "compute_temporal_training_stats",
]
