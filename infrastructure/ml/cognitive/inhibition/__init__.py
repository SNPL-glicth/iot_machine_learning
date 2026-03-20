"""Inhibition subsystem — weight suppression for unstable engines.

Components:
    - InhibitionGate: Engine weight suppression logic
    - InhibitionConfig: Configuration dataclass
    - InhibitionRules: Rule-based inhibition logic
"""

from .gate import InhibitionGate, InhibitionConfig
from .rules import apply_inhibition_rules

__all__ = [
    "InhibitionGate",
    "InhibitionConfig",
    "apply_inhibition_rules",
]
