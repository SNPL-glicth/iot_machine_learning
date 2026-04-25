"""Motores de predicción — implementaciones concretas de PredictionPort.

Auto-registro de motores al importar el paquete.

Reorganizado 2026-03-20:
- core/ — EngineFactory, register_engine, discover_engines
- baseline/ — BaselineMovingAverageEngine, predict_moving_average
- taylor/ — TaylorPredictionEngine (+ math modules)
- statistical/ — StatisticalPredictionEngine  
- ensemble/ — EnsembleWeightedPredictor

To add a new engine:
1. Create a new module in this package.
2. Either use ``@register_engine("name")`` decorator, or
3. Add ``EngineFactory.register(...)`` here.

Use ``discover_engines(package_path)`` for plugin auto-discovery.
"""

from iot_machine_learning.infrastructure.ml.engines.core import (
    EngineFactory,
    discover_engines,
    register_engine,
    BaselineMovingAverageEngine,
)
from iot_machine_learning.infrastructure.ml.engines.taylor import TaylorPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.statistical import StatisticalPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.deprecated.ensemble_predictor import EnsembleWeightedPredictor

# BaselineMovingAverageEngine ya se registra en core/factory.py
EngineFactory.register("taylor", TaylorPredictionEngine)
EngineFactory.register("statistical", StatisticalPredictionEngine)

__all__ = [
    "EngineFactory",
    "register_engine",
    "discover_engines",
    "BaselineMovingAverageEngine",
    "TaylorPredictionEngine",
    "StatisticalPredictionEngine",
    "EnsembleWeightedPredictor",
]
