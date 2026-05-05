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
from ..detectors.multivariate_detector import MultivariateDetector


def create_default_detectors(
    config: AnomalyDetectorConfig,
) -> List[SubDetector]:
    """Create the default ensemble of sub-detectors.

    This factory function extracts the hardcoded detector list so it can
    be reused, overridden, or extended.  Pass the result to
    ``VotingAnomalyDetector(sub_detectors=...)`` to customize.

    Args:
        config: Anomaly detector configuration.

    Returns:
        List of sub-detectors (9 if multivariate enabled, else 8).
    """
    detectors = [
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
    
    # FASE 3: Add multivariate detector if enabled
    if getattr(config, 'enable_multivariate', False):
        detectors.append(
            MultivariateDetector(
                min_series=getattr(config, 'multivariate_min_series', 3),
                pca_components=getattr(config, 'multivariate_pca_components', 2),
                baseline_percentile=getattr(config, 'multivariate_baseline_percentile', 95.0),
                warmup_samples=getattr(config, 'multivariate_warmup_samples', 30),
                enabled=True,
            )
        )
    
    return detectors
