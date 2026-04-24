"""Plasticity entities for adaptive learning system.

This package contains domain entities for the advanced plasticity system:
- SignalContext: Contextual information for adaptive learning
- EnginePlasticityState: State tracking for individual engines
"""

from .signal_context import SignalContext
from .engine_plasticity_state import EnginePlasticityState

__all__ = [
    "SignalContext",
    "EnginePlasticityState",
]
