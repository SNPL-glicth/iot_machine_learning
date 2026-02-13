"""Re-exports para backward compatibility.

Los módulos individuales ahora viven en:
- ``training_stats.py`` → TrainingStats, compute_training_stats
- ``scoring_functions.py`` → compute_z_score, compute_z_vote, compute_iqr_*,
                              weighted_vote, compute_consensus_confidence
- ``temporal_stats.py`` → TemporalTrainingStats, compute_temporal_training_stats
"""

from .training_stats import TrainingStats, compute_training_stats
from .scoring_functions import (
    compute_z_score,
    compute_z_vote,
    compute_iqr_bounds,
    compute_iqr_vote,
    weighted_vote,
    compute_consensus_confidence,
)
from .temporal_stats import (
    TemporalTrainingStats,
    compute_temporal_training_stats,
    _compute_distribution_stats,
    _empty_dist,
)

__all__ = [
    "TrainingStats",
    "compute_training_stats",
    "compute_z_score",
    "compute_z_vote",
    "compute_iqr_bounds",
    "compute_iqr_vote",
    "weighted_vote",
    "compute_consensus_confidence",
    "TemporalTrainingStats",
    "compute_temporal_training_stats",
]
