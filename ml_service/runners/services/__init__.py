"""Services for ML online processing."""

try:
    from .window_analyzer import WindowAnalyzer
except (ImportError, ModuleNotFoundError):
    WindowAnalyzer = None  # type: ignore[assignment]

try:
    from .threshold_validator import ThresholdValidator
except (ImportError, ModuleNotFoundError):
    ThresholdValidator = None  # type: ignore[assignment]

try:
    from .event_persister import MLEventPersister
except (ImportError, ModuleNotFoundError):
    MLEventPersister = None  # type: ignore[assignment]

try:
    from .explanation_builder import ExplanationBuilder
except (ImportError, ModuleNotFoundError):
    ExplanationBuilder = None  # type: ignore[assignment]

__all__ = [
    "WindowAnalyzer",
    "ThresholdValidator",
    "MLEventPersister",
    "ExplanationBuilder",
]
