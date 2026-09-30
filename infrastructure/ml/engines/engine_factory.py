"""Backward-compatibility module for engine factory.

Re-exports EngineFactory, discover_engines, and register_engine from
infrastructure.ml.engines.core.factory for backwards compatibility with
existing callers and test suites.
"""

from __future__ import annotations

from .core.factory import (
    EngineFactory,
    discover_engines,
    register_engine,
)

__all__ = [
    "EngineFactory",
    "discover_engines",
    "register_engine",
]
