"""Gradient-based optimization algorithms."""

from .sgd import SGDOptimizer, MomentumSGD, NesterovSGD
from .gradient_clip import clip_gradients, compute_gradient_norm
from .scheduler import (
    LearningRateScheduler,
    StepLRScheduler,
    CosineAnnealingScheduler,
    WarmupScheduler,
)

# NOTE: Adam, AdaGrad, RMSProp moved to _experimental/gradient/adam.py
# They are not currently used in production pipelines.

__all__ = [
    "SGDOptimizer",
    "MomentumSGD",
    "NesterovSGD",
    "clip_gradients",
    "compute_gradient_norm",
    "LearningRateScheduler",
    "StepLRScheduler",
    "CosineAnnealingScheduler",
    "WarmupScheduler",
]
