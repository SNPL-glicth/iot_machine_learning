"""Plasticity subsystem — regime-contextual weight learning.

Two modes:
1. Base Plasticity (PlasticityTracker): Simple regime-based EMA
2. Advanced Plasticity: Context-aware 4-component system

Components:
    - PlasticityTracker: Base regime-based weight learning (EMA)
    - AdaptiveLearningRate: Context-aware learning rates
    - AdvancedPlasticityCoordinator: Coordinates 4 advanced components
    - ContextualPlasticityTracker: MAE tracking by context
    - build_advanced_plasticity: Factory for advanced system

Modular Components (for advanced usage):
    - constants: Configuration constants and PlasticityConfig
    - redis_client: Redis operations with scope support
    - persistence: SQL persistence operations
    - checkpoint: Export/import for gossip protocol
"""

from .base import PlasticityTracker
from .adaptive_learning_rate import AdaptiveLearningRate
from .advanced_plasticity_coordinator import AdvancedPlasticityCoordinator
from .contextual_plasticity_tracker import ContextualPlasticityTracker
from .factory import build_advanced_plasticity, null_advanced_plasticity

# Modular components (advanced usage)
from .constants import PlasticityConfig
from .redis_client import PlasticityRedisClient
from .persistence import PlasticityPersistence
from .checkpoint import PlasticityCheckpoint
from .lr_calculator import compute_learning_rate, get_regime_factor
from .weight_calculator import compute_weights_from_accuracy
from .contextual_storage import ContextualErrorStorage
from .context_builder import build_plasticity_context
from .error_persister import ErrorPersister

__all__ = [
    # Core
    "PlasticityTracker",
    "AdaptiveLearningRate",
    "AdvancedPlasticityCoordinator",
    "ContextualPlasticityTracker",
    "build_advanced_plasticity",
    "null_advanced_plasticity",
    # Modular components
    "PlasticityConfig",
    "PlasticityRedisClient",
    "PlasticityPersistence",
    "PlasticityCheckpoint",
    "compute_learning_rate",
    "get_regime_factor",
    "compute_weights_from_accuracy",
    "ContextualErrorStorage",
    "build_plasticity_context",
    "ErrorPersister",
]
