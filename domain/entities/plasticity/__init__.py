"""Plasticity entities for adaptive learning system.

This package contains domain entities for the advanced plasticity system:
- PlasticityContext: Contextual information for adaptive learning
- EnginePlasticityState: State tracking for individual engines
"""

from .plasticity_context import PlasticityContext
from .engine_plasticity_state import EnginePlasticityState

__all__ = [
    "PlasticityContext",
    "EnginePlasticityState",
]
