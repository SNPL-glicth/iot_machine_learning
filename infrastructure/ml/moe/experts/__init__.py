"""Expert adapters for MoE architecture.

Wraps existing PredictionEngines as ExpertPort implementations.

Exports:
- BaselineExpert: Adapter for BaselineMovingAverageEngine
- StatisticalExpert: Adapter for StatisticalPredictionEngine  
- TaylorExpert: Adapter for TaylorPredictionEngine
"""

from .baseline_expert import BaselineExpert
from .statistical_expert import StatisticalExpert
from .taylor_expert import TaylorExpert

__all__ = [
    "BaselineExpert",
    "StatisticalExpert", 
    "TaylorExpert",
]
