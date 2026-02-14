"""ML Runners module - Procesadores online y batch.

Estructura modular:
- models/: Dataclasses (SensorState, OnlineAnalysis)
- services/: Servicios especializados (WindowAnalyzer, ThresholdValidator, etc.)
- ml_stream_runner.py: Runner modular (~300 líneas)
- bridge_config/: Feature flag routing for enterprise bridge
- adapters/: Enterprise prediction adapter
- wiring/: DI container for batch enterprise bridge
- monitoring/: A/B metrics and audit logging

NOTE: Imports are lazy to avoid blocking subpackages when
ml_stream_runner has unresolved cross-service dependencies.
"""


def __getattr__(name: str):
    """Lazy imports to avoid blocking subpackage access."""
    _stream_exports = {
        "SimpleMlOnlineProcessor", "run_stream", "main",
    }
    _model_exports = {"SensorState", "OnlineAnalysis"}
    _service_exports = {
        "WindowAnalyzer", "ThresholdValidator",
        "ExplanationBuilder", "MLEventPersister",
    }

    if name in _stream_exports:
        from .ml_stream_runner import SimpleMlOnlineProcessor, run_stream, main as _main
        return {"SimpleMlOnlineProcessor": SimpleMlOnlineProcessor, "run_stream": run_stream, "main": _main}[name]

    if name in _model_exports:
        from .models import SensorState, OnlineAnalysis
        return {"SensorState": SensorState, "OnlineAnalysis": OnlineAnalysis}[name]

    if name in _service_exports:
        from .services import WindowAnalyzer, ThresholdValidator, ExplanationBuilder, MLEventPersister
        return {
            "WindowAnalyzer": WindowAnalyzer, "ThresholdValidator": ThresholdValidator,
            "ExplanationBuilder": ExplanationBuilder, "MLEventPersister": MLEventPersister,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
