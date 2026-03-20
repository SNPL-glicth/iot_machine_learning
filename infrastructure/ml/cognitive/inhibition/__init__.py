"""Inhibition subsystem — weight suppression for unstable engines.

Components:
    - InhibitionGate: Engine weight suppression logic
    - InhibitionConfig: Configuration dataclass
    - InhibitionState: Inhibition state tracking
"""

from .gate import InhibitionGate, InhibitionConfig, InhibitionState

__all__ = [
    "InhibitionGate",
    "InhibitionState",
    "InhibitionConfig",
]
