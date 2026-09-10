"""Motores de predicción — implementaciones concretas de PredictionPort.

Auto-registro de motores al importar el paquete.

Reorganizado 2026-08-22:
- core/ — EngineFactory, register_engine, discover_engines
- baseline/ — BaselineMovingAverageEngine, predict_moving_average
- taylor/ — TaylorPredictionEngine (+ math modules)
- statistical/ — StatisticalPredictionEngine
- kalman/ — KalmanPredictionEngine
- multivariate/ — MultivariateEngine
- seasonal/ — SeasonalPredictorEngine
"""

from .core import (
    EngineFactory,
    discover_engines,
    register_engine,
    BaselineMovingAverageEngine,
)
from .taylor import TaylorPredictionEngine
from .statistical import StatisticalPredictionEngine
from .kalman import KalmanPredictionEngine
from .rosa_roja import RosaRojaMoEEngine

# BaselineMovingAverageEngine ya se registra en core/factory.py
EngineFactory.register("taylor", TaylorPredictionEngine)
EngineFactory.register("statistical", StatisticalPredictionEngine)
EngineFactory.register("kalman", KalmanPredictionEngine)
EngineFactory.register("rosa_roja", RosaRojaMoEEngine)

__all__ = [
    "EngineFactory",
    "register_engine",
    "discover_engines",
    "BaselineMovingAverageEngine",
    "TaylorPredictionEngine",
    "StatisticalPredictionEngine",
    "KalmanPredictionEngine",
    "RosaRojaMoEEngine",
]