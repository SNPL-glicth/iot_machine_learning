"""Registry module for MoE expert management.

Exports:
- ExpertRegistry: Main registry class
- ExpertCapability: Capability declaration dataclass
- Expert: Protocol for registerable experts
"""

from .expert_capability import ExpertCapability
from .expert_registry import ExpertRegistry, Expert, ExpertEntry

__all__ = [
    "ExpertCapability",
    "ExpertRegistry",
    "Expert",
    "ExpertEntry",
]
