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
# LightGBM movido a _experimental/ — ver docs/ENGINES.md
# from iot_machine_learning.infrastructure.ml.engines.lightgbm import LightGBMPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.adaptive_ensemble import (
    AdaptiveEnsembleEngine,
)
from iot_machine_learning.infrastructure.ml.engines.kalman import (
    KalmanPredictionEngine,
)
# from iot_machine_learning.infrastructure.ml.engines.deprecated.ensemble_predictor import EnsembleWeightedPredictor

# BaselineMovingAverageEngine ya se registra en core/factory.py
EngineFactory.register("taylor", TaylorPredictionEngine)
EngineFactory.register("statistical", StatisticalPredictionEngine)
# LightGBM movido a _experimental/ — ver docs/ENGINES.md
# EngineFactory.register("lightgbm", LightGBMPredictionEngine)
# AdaptiveEnsembleEngine auto-registers via @register_engine
EngineFactory.register("adaptive_ensemble", AdaptiveEnsembleEngine)
EngineFactory.register("kalman", KalmanPredictionEngine)

__all__ = [
    "EngineFactory",
    "register_engine",
    "discover_engines",
    "BaselineMovingAverageEngine",
    "TaylorPredictionEngine",
    "StatisticalPredictionEngine",
    # "LightGBMPredictionEngine",  # movido a _experimental/
    "AdaptiveEnsembleEngine",
    "KalmanPredictionEngine",
    # "EnsembleWeightedPredictor",
]
