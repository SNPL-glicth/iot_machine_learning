"""Pipeline Phases — MED-1 Refactoring.

Modular pipeline phases using Strategy pattern.
"""

from __future__ import annotations

from .protocols import PipelinePhase, PhaseMetadata, PipelineResultMetadata
from .context import PipelineContext, create_initial_context

__all__ = [
    "PipelinePhase",
    "PhaseMetadata", 
    "PipelineResultMetadata",
    "PipelineContext",
    "create_initial_context",
]
