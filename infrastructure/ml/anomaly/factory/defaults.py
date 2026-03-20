"""Factory for the default ensemble of anomaly sub-detectors.

Extracted from voting_anomaly_detector.py so the factory function can be
imported and reused independently of the VotingAnomalyDetector class.

Backward-compatible: voting_anomaly_detector still re-exports
create_default_detectors from here.
"""

from __future__ import annotations

from typing import List

from ..core.config import AnomalyDetectorConfig
from ..core.protocol import SubDetector
from ..detectors import (
    AccelerationZDetector,
    IQRDetector,
    IsolationForestDetector,
    LOFDetector,
    VelocityZDetector,
    ZScoreDetector,
)
from ..detectors.isolation_forest_detector import IsolationForestNDDetector
from ..detectors.lof_detector import LOFNDDetector


def create_default_detectors(
    config: AnomalyDetectorConfig,
) -> List[SubDetector]:
    """Create the default ensemble of 8 sub-detectors.

    This factory function extracts the hardcoded detector list so it can
    be reused, overridden, or extended.  Pass the result to
    ``VotingAnomalyDetector(sub_detectors=...)`` to customize.

    Args:
        config: Anomaly detector configuration.

    Returns:
        List of 8 default sub-detectors.
    """
    return [
        ZScoreDetector(
            lower=config.z_vote_lower,
            upper=config.z_vote_upper,
        ),
        IQRDetector(),
        IsolationForestDetector(
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            random_state=config.random_state,
        ),
        LOFDetector(
            contamination=config.contamination,
            max_neighbors=config.lof_max_neighbors,
        ),
        VelocityZDetector(
            lower=config.z_vote_lower,
            upper=config.z_vote_upper,
        ),
        AccelerationZDetector(
            lower=config.z_vote_lower,
            upper=config.z_vote_upper,
        ),
        IsolationForestNDDetector(
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            min_training_points=config.min_training_points,
        ),
        LOFNDDetector(
            contamination=config.contamination,
            max_neighbors=config.lof_max_neighbors,
            min_training_points=config.min_training_points,
        ),
    ]
