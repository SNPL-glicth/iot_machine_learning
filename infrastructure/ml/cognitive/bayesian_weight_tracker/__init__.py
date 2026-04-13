"""Bayesian Weight Tracker — regime-contextual weight learning via Bayesian inference.

Uses Bayesian inference (Gaussian priors) to track per-regime engine accuracy.
NOT reinforcement learning or neural plasticity — honest naming for honest code.

Two modes:
1. Base Tracker (BayesianWeightTracker): Simple regime-based EMA with Bayesian updates
2. Advanced Tracker: Context-aware 4-component system

Components:
    - BayesianWeightTracker: Base regime-based weight learning (Bayesian EMA)
    - AdaptiveLearningRate: Context-aware learning rates
    - AdvancedBayesianCoordinator: Coordinates 4 advanced components
    - ContextualWeightTracker: MAE tracking by context
    - build_advanced_bayesian: Factory for advanced system

Modular Components (for advanced usage):
    - constants: Configuration constants and WeightTrackerConfig
    - redis_client: Redis operations with scope support
    - persistence: SQL persistence operations
    - checkpoint: Export/import for gossip protocol
"""

from .base import BayesianWeightTracker
from .adaptive_learning_rate import AdaptiveLearningRate
from .advanced_bayesian_coordinator import AdvancedBayesianCoordinator
from .contextual_weight_tracker import ContextualWeightTracker
from .factory import build_advanced_bayesian, null_advanced_bayesian

# Modular components (advanced usage)
from .constants import WeightTrackerConfig
from .redis_client import WeightTrackerRedisClient
from .persistence import WeightTrackerPersistence
from .checkpoint import WeightTrackerCheckpoint
from .lr_calculator import compute_learning_rate, get_regime_factor
from .weight_calculator import compute_weights_from_accuracy
from .contextual_storage import ContextualErrorStorage
from .context_builder import build_plasticity_context
from .error_persister import ErrorPersister

__all__ = [
    # Core
    "BayesianWeightTracker",
    "AdaptiveLearningRate",
    "AdvancedBayesianCoordinator",
    "ContextualWeightTracker",
    "build_advanced_bayesian",
    "null_advanced_bayesian",
    # Modular components
    "WeightTrackerConfig",
    "WeightTrackerRedisClient",
    "WeightTrackerPersistence",
    "WeightTrackerCheckpoint",
    "compute_learning_rate",
    "get_regime_factor",
    "compute_weights_from_accuracy",
    "ContextualErrorStorage",
    "build_weight_tracker_context",
    "ErrorPersister",
]

# Backward compatibility aliases (deprecated, will be removed in Phase 3)
PlasticityTracker = BayesianWeightTracker
PlasticityConfig = WeightTrackerConfig
PlasticityRedisClient = WeightTrackerRedisClient
PlasticityPersistence = WeightTrackerPersistence
PlasticityCheckpoint = WeightTrackerCheckpoint
build_advanced_plasticity = build_advanced_bayesian
null_advanced_plasticity = null_advanced_bayesian
build_weight_tracker_context = build_plasticity_context  # backward compat alias
