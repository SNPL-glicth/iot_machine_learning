"""Gating strategies for MoE architecture.

Exporta estrategias de routing disponibles:
- GatingNetwork: Base class (Strategy pattern)
- RegimeBasedGating: Heurístico por régimen (Fase 1)
- TreeGatingNetwork: XGBoost con SHAP (Fase 2)
- ContextVector, GatingProbs: Tipos de datos
"""

from .base import GatingNetwork, GatingProbs, ContextVector
from .regime_based import RegimeBasedGating, RegimeRoutingRule
from .tree_gating import TreeGatingNetwork, TreeRoutingExplanation

__all__ = [
    "GatingNetwork",
    "GatingProbs",
    "ContextVector",
    "RegimeBasedGating",
    "RegimeRoutingRule",
    "TreeGatingNetwork",
    "TreeRoutingExplanation",
]