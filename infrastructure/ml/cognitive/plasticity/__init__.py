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
"""

from .base import PlasticityTracker
from .adaptive_learning_rate import AdaptiveLearningRate
from .advanced_plasticity_coordinator import AdvancedPlasticityCoordinator
from .contextual_plasticity_tracker import ContextualPlasticityTracker
from .factory import build_advanced_plasticity, null_advanced_plasticity

__all__ = [
    "PlasticityTracker",
    "AdaptiveLearningRate",
    "AdvancedPlasticityCoordinator",
    "ContextualPlasticityTracker",
    "build_advanced_plasticity",
    "null_advanced_plasticity",
]
