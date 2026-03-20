"""Core engine infrastructure — factory + registry + auto-discovery.

Components:
    - EngineFactory: Central registry for all prediction engines
    - register_engine: Decorator for auto-registering engines
    - discover_engines: Plugin discovery via package scanning
    - BaselineMovingAverageEngine: Built-in fallback engine
"""

from .factory import (
    BaselineMovingAverageEngine,
    EngineFactory,
    discover_engines,
    register_engine,
)

__all__ = [
    "EngineFactory",
    "register_engine",
    "discover_engines",
    "BaselineMovingAverageEngine",
]
