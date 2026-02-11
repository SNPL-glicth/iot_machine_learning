"""Motores de predicción — implementaciones concretas de PredictionPort.

Auto-registro de motores al importar el paquete.
"""

from iot_machine_learning.infrastructure.ml.engines.engine_factory import EngineFactory
from iot_machine_learning.infrastructure.ml.engines.taylor_engine import TaylorPredictionEngine

# BaselineMovingAverageEngine ya se registra en engine_factory.py
EngineFactory.register("taylor", TaylorPredictionEngine)
