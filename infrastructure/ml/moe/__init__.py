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
- CapacityScheduler: Adaptación dinámica de carga

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
from .gateway import MoEGateway, MoEMetadata
from .fusion import SparseFusionLayer, FusedResult
from .scheduler import CapacityScheduler, SystemLoadMetrics, create_default_scheduler

# Gating strategies
from .gating import (
    GatingNetwork,
    GatingProbs,
    ContextVector,
    RegimeBasedGating,
    RegimeRoutingRule,
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
    "FusedResult",
    "CapacityScheduler",
    "SystemLoadMetrics",
    "create_default_scheduler",
    # Gating
    "GatingNetwork",
    "GatingProbs",
    "ContextVector",
    "RegimeBasedGating",
    "RegimeRoutingRule",
    # Adapters
    "EngineExpertAdapter",
    "EnsembleExpertAdapter",
    "create_baseline_expert",
    "create_statistical_expert",
    "create_taylor_expert",
]

__version__ = "1.0.0"
