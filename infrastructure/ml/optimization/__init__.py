"""Optimization algorithms — gradient descent + convex/non-convex methods.

Gradient-based optimizers:
- SGD, Momentum, Nesterov
- Adam, AdaGrad, RMSProp
- Learning rate schedulers

Convex optimization:
- Newton-Raphson, L-BFGS
- Proximal gradient (L1/L2 regularization)
- Conjugate gradient

Non-convex optimization:
- Simulated annealing
- Genetic algorithms
- Particle swarm optimization

Unified optimizer:
- Auto-selects best method per problem
"""

from .types import OptimizationResult, OptimizerConfig
from .gradient import SGDOptimizer, AdamOptimizer, MomentumSGD
from .convex import NewtonRaphsonOptimizer, LBFGSOptimizer
from .nonconvex import SimulatedAnnealing, GeneticOptimizer, ParticleSwarmOptimizer
from .unified import UnifiedOptimizer

__all__ = [
    "OptimizationResult",
    "OptimizerConfig",
    "SGDOptimizer",
    "AdamOptimizer",
    "MomentumSGD",
    "NewtonRaphsonOptimizer",
    "LBFGSOptimizer",
    "SimulatedAnnealing",
    "GeneticOptimizer",
    "ParticleSwarmOptimizer",
    "UnifiedOptimizer",
]
