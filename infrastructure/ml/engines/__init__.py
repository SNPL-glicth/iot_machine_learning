"""Motores de predicción — implementaciones concretas de PredictionPort.

Auto-registro de motores al importar el paquete.

To add a new engine:
1. Create a new module in this package.
2. Either use ``@register_engine("name")`` decorator, or
3. Add ``EngineFactory.register(...)`` here.

Use ``discover_engines(package_path)`` for plugin auto-discovery.
"""

from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
    EngineFactory,
    discover_engines,
    register_engine,
)
from iot_machine_learning.infrastructure.ml.engines.taylor_engine import TaylorPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.statistical_engine import StatisticalPredictionEngine

# BaselineMovingAverageEngine ya se registra en engine_factory.py
EngineFactory.register("taylor", TaylorPredictionEngine)
EngineFactory.register("statistical", StatisticalPredictionEngine)
