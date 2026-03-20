"""Fusion subsystem — weighted fusion and engine selection.

Components:
    - WeightedFusion: Multi-engine fusion logic
    - FusionPhases: Phase orchestration helpers
    - WeightMediator: Weight conflict resolution
    - WeightAdjustmentService: Dynamic weight adjustment
    - ContextualWeightCalculator: Context-aware weight computation
"""

from .engine_selector import WeightedFusion
from .weight_mediator import WeightMediator
from .weight_adjustment_service import WeightAdjustmentService

__all__ = [
    "WeightedFusion",
    "WeightMediator",
    "WeightAdjustmentService",
]
