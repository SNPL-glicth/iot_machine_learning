"""Mixture of Experts (MoE) infrastructure for ZENIN.

Implementa arquitectura MoE siguiendo principios:
- SOLID: Ports/Adapters, Strategy, Factory
- Clean Code: Funciones pequeñas, nombres descriptivos
- ISO 25010: Performance, reliability, maintainability
- ISO 42001: AI governance, explicabilidad

Componentes principales:
- MoEGateway: Implementa PredictionPort con MoE interno
- ExpertRegistry: Catálogo de expertos disponibles
- GatingNetwork: Estrategias de routing (Strategy pattern)
- SparseFusionLayer: Fusión de k expertos (no todos)
- FusionWeights: Pesos normalizados de fusión
- RolloutDecider: Rollout gradual por sensor_id hash

Ejemplo de uso:
    >>> from infrastructure.ml.moe import MoEGateway, ExpertRegistry
    >>> from infrastructure.ml.moe.gating import RegimeBasedGating
    >>> 
    >>> # Setup
    >>> registry = ExpertRegistry()
    >>> registry.register("baseline", baseline_expert, baseline_caps)
    >>> 
    >>> gating = RegimeBasedGating.with_default_rules(["baseline"])
    >>> gateway = MoEGateway(registry, gating, SparseFusionLayer())
    >>> 
    >>> # Uso
    >>> prediction = gateway.predict(sensor_window)
    >>> print(prediction.metadata["moe"]["selected_experts"])
"""

from .registry import ExpertRegistry, ExpertEntry
from .gateway.moe_gateway import MoEGateway, MoEMetadata
from .fusion.sparse_fusion import SparseFusionLayer, FusionWeights
from .rollout.rollout_decider import RolloutDecider
from .rollout.rollout_bridge import RolloutPredictionPortBridge
from .metrics.moe_alert_service import MoEAlertService
from .feature_context import FeatureContext
from .gating.strategy import GatingStrategy
from .gating.contextual_regime import ContextualRegimeGating
from .fusion.discrepancy_aware import DiscrepancyAwareFusion
from .engine.moe_prediction_engine import MoEPredictionEngine
from .ab.moe_ab_logger import MoEABLogger, ABLogEntry

# Gating strategies
from .gating import (
    GatingNetwork,
    GatingProbs,
    RegimeBasedGating,
    RegimeRoutingRule,
    TreeGatingNetwork,
)

# Expert adapters
from .expert_wrappers.engine_adapter import (
    EngineExpertAdapter,
    EnsembleExpertAdapter,
    create_baseline_expert,
    create_statistical_expert,
    create_taylor_expert,
)

__all__ = [
    # Core
    "MoEGateway",
    "MoEMetadata",
    "ExpertRegistry",
    "ExpertEntry",
    "SparseFusionLayer",
    "FusionWeights",
    "FeatureContext",
    # Rollout
    "RolloutDecider",
    "RolloutPredictionPortBridge",
    # Metrics
    "MoEAlertService",
    # Gating
    "GatingNetwork",
    "GatingProbs",
    "RegimeBasedGating",
    "RegimeRoutingRule",
    "TreeGatingNetwork",
    "GatingStrategy",
    "ContextualRegimeGating",
    # Fusion
    "DiscrepancyAwareFusion",
    # Engine
    "MoEPredictionEngine",
    # A/B
    "MoEABLogger",
    "ABLogEntry",
    # Adapters
    "EngineExpertAdapter",
    "EnsembleExpertAdapter",
    "create_baseline_expert",
    "create_statistical_expert",
    "create_taylor_expert",
]

__version__ = "1.0.0"
