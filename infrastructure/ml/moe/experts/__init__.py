"""Expert adapters for MoE architecture.

Wraps existing PredictionEngines as ExpertPort implementations.
NeuralExpert is a standalone numpy-based neural network expert.

Exports:
- BaselineExpert: Adapter for BaselineMovingAverageEngine
- StatisticalExpert: Adapter for StatisticalPredictionEngine  
- TaylorExpert: Adapter for TaylorPredictionEngine
- NeuralExpert: Tiny feedforward NN trained per-series
"""

from .baseline_expert import BaselineExpert
from .statistical_expert import StatisticalExpert
from .taylor_expert import TaylorExpert
from .neural_expert import NeuralExpert, create_neural_expert

__all__ = [
    "BaselineExpert",
    "StatisticalExpert", 
    "TaylorExpert",
    "NeuralExpert",
    "create_neural_expert",
]
