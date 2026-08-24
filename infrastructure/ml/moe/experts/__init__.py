"""Expert adapters for MoE architecture.

Wraps existing PredictionEngines as ExpertPort implementations.
NeuralExpert is a standalone numpy-based neural network expert.
RosaRojaExpert: Adapter for Rosa Roja Core (trajectory/rhythm reasoning).

Exports:
- BaselineExpert: Adapter for BaselineMovingAverageEngine
- StatisticalExpert: Adapter for StatisticalPredictionEngine  
- TaylorExpert: Adapter for TaylorPredictionEngine
- NeuralExpert: Tiny feedforward NN trained per-series
- RosaRojaExpert: Adapter for Rosa Roja Core (challenger, feature-flagged)
"""

from .baseline_expert import BaselineExpert
from .statistical_expert import StatisticalExpert
from .taylor_expert import TaylorExpert
from .neural_expert import NeuralExpert, create_neural_expert
from .rosa_roja_expert import RosaRojaExpert, RosaRojaResult

__all__ = [
    "BaselineExpert",
    "StatisticalExpert", 
    "TaylorExpert",
    "NeuralExpert",
    "create_neural_expert",
    "RosaRojaExpert",
    "RosaRojaResult",
]
