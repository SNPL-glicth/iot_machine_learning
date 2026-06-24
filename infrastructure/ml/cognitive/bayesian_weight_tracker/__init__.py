"""Bayesian Weight Tracker — consolidated to 5 files.

Core files:
  config.py        — BayesianWeightConfig, WeightTrackerConfig, constants
  updater.py       — GaussianPrior, BayesianUpdater, VarianceEstimator, accuracy, L2
  persistence.py   — WeightTrackerPersistence, WeightTrackerRedisClient, WeightTrackerCheckpoint
  drift_response.py— GradualDriftResponse
  tracker.py       — BayesianWeightTracker (main class)

Backward-compatible exports maintained for all previous public symbols.
"""

from .tracker import BayesianWeightTracker
from .updater import (
    GaussianPrior,
    BayesianUpdater,
    VarianceEstimator,
    compute_accuracy,
    compute_regularization_strength,
    apply_l2_regularization,
    compute_weights_from_accuracy,
    build_regime_key,
    build_fallback_key,
    should_use_per_sensor,
)
from .persistence import (
    WeightTrackerPersistence,
    WeightTrackerRedisClient,
    WeightTrackerCheckpoint,
)
from .config import BayesianWeightConfig, WeightTrackerConfig
from .drift_response import GradualDriftResponse

# Advanced tracker components (kept as standalone modules)
from .adaptive_learning_rate import AdaptiveLearningRate
from .advanced_bayesian_coordinator import AdvancedBayesianCoordinator
from .contextual_weight_tracker import ContextualWeightTracker
from .factory import build_advanced_bayesian, null_advanced_bayesian

# Backward-compatible aliases from old architecture
PlasticityTracker = BayesianWeightTracker
PlasticityConfig = WeightTrackerConfig
PlasticityRedisClient = WeightTrackerRedisClient
PlasticityPersistence = WeightTrackerPersistence
PlasticityCheckpoint = WeightTrackerCheckpoint
build_advanced_plasticity = build_advanced_bayesian
null_advanced_plasticity = null_advanced_bayesian

__all__ = [
    "BayesianWeightTracker",
    "BayesianWeightConfig",
    "WeightTrackerConfig",
    "WeightTrackerRedisClient",
    "WeightTrackerPersistence",
    "WeightTrackerCheckpoint",
    "GradualDriftResponse",
    "GaussianPrior",
    "BayesianUpdater",
    "VarianceEstimator",
    "AdaptiveLearningRate",
    "AdvancedBayesianCoordinator",
    "ContextualWeightTracker",
    "build_advanced_bayesian",
    "null_advanced_bayesian",
    "compute_accuracy",
    "compute_regularization_strength",
    "apply_l2_regularization",
    "compute_weights_from_accuracy",
    "build_regime_key",
    "build_fallback_key",
    "should_use_per_sensor",
    "PlasticityTracker",
    "PlasticityConfig",
    "PlasticityRedisClient",
    "PlasticityPersistence",
    "PlasticityCheckpoint",
    "build_advanced_plasticity",
    "null_advanced_plasticity",
]
