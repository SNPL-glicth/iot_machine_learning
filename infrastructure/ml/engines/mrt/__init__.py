"""MRT (Maxwell-Ramanujan-Tesla) Conjugate Engine Package (ZENIN v2.4+).

Provides first-class inverse kinematics prediction engine and phase conjugator algorithms:
- MRTEngine: PredictionEngine implementing the conjugate negative pole z2.
- algorithms: MaxwellCurlField, RamanujanCrystal, HopfSpinorField, MRTPipeline.
"""

from __future__ import annotations

from .mrt_engine import MRTEngine
from .phase_conjugator import PhaseConjugator
from .algorithms import (
    MaxwellCurlField,
    RamanujanCrystal,
    HopfSpinorField,
    MRTPipeline,
    MRTExecutionResult,
)

__all__ = [
    "MRTEngine",
    "PhaseConjugator",
    "MaxwellCurlField",
    "RamanujanCrystal",
    "HopfSpinorField",
    "MRTPipeline",
    "MRTExecutionResult",
]
