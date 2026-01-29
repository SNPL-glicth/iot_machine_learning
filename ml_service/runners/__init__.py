"""ML Runners module - Procesadores online y batch.

Estructura modular:
- models/: Dataclasses (SensorState, OnlineAnalysis)
- services/: Servicios especializados (WindowAnalyzer, ThresholdValidator, etc.)
- ml_stream_runner.py: Runner modular (~300 líneas)
"""

from .ml_stream_runner import (
    SimpleMlOnlineProcessor,
    run_stream,
    main,
)
from .models import SensorState, OnlineAnalysis
from .services import (
    WindowAnalyzer,
    ThresholdValidator,
    ExplanationBuilder,
    MLEventPersister,
)

__all__ = [
    "SimpleMlOnlineProcessor",
    "run_stream",
    "main",
    "SensorState",
    "OnlineAnalysis",
    "WindowAnalyzer",
    "ThresholdValidator",
    "ExplanationBuilder",
    "MLEventPersister",
]
