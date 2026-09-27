"""MRT Algorithmic Core Package (ZENIN v2.4+).

Conjugate mirror of Rosa Roja algorithms:
- modules: MaxwellCurlField, RamanujanCrystal, HopfSpinorField.
- pipeline: MRTPipeline, MRTExecutionResult.
"""

from __future__ import annotations

from .modules.maxwell_curl_field import MaxwellCurlField
from .modules.ramanujan_crystal import RamanujanCrystal
from .modules.hopf_spinor_field import HopfSpinorField, SpinorPoleComponents
from .mrt_pipeline import MRTPipeline, MRTExecutionResult

__all__ = [
    "MaxwellCurlField",
    "RamanujanCrystal",
    "HopfSpinorField",
    "SpinorPoleComponents",
    "MRTPipeline",
    "MRTExecutionResult",
]
