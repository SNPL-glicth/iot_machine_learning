"""Optimization algorithms — gradient descent + convex/non-convex methods.

Gradient-based optimizers:
- SGD, Momentum, Nesterov
- Learning rate schedulers
- NOTE: Adam, AdaGrad, RMSProp moved to _experimental/ (not production)

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
from .gradient import SGDOptimizer, MomentumSGD
from .convex import NewtonRaphsonOptimizer, LBFGSOptimizer
from .nonconvex import SimulatedAnnealing, GeneticOptimizer, ParticleSwarmOptimizer
from .unified import UnifiedOptimizer

__all__ = [
    "OptimizationResult",
    "OptimizerConfig",
    "SGDOptimizer",
    "MomentumSGD",
    "NewtonRaphsonOptimizer",
    "LBFGSOptimizer",
    "SimulatedAnnealing",
    "GeneticOptimizer",
    "ParticleSwarmOptimizer",
    "UnifiedOptimizer",
]
