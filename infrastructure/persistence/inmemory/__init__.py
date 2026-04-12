"""In-memory persistence implementations.

Used for development, testing, and fallback scenarios.
"""

from __future__ import annotations

from iot_machine_learning.infrastructure.persistence.inmemory.plasticity_repository import (
    InMemoryPlasticityRepository,
)

__all__ = ["InMemoryPlasticityRepository"]
