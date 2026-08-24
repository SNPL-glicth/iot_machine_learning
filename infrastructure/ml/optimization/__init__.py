"""Optimization algorithms — gradient descent methods.

Gradient-based optimizers:
- SGD, Momentum, Nesterov
- Learning rate schedulers
"""

from .types import OptimizationResult, OptimizerConfig
from .gradient import SGDOptimizer, MomentumSGD

__all__ = [
    "OptimizationResult",
    "OptimizerConfig",
    "SGDOptimizer",
    "MomentumSGD",
]