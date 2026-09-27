"""MRT Algorithmic Modules Package (ZENIN v2.4+).

Exposes core conjugate field operators:
- MaxwellCurlField: Rotational vorticity ‖∇ × B‖.
- RamanujanCrystal: 4D augmented symplectic tensor Tr(J*).
- HopfSpinorField: Rational algebraic spinor field in ℂ².
"""

from __future__ import annotations

from .maxwell_curl_field import MaxwellCurlField
from .ramanujan_crystal import RamanujanCrystal
from .hopf_spinor_field import HopfSpinorField, SpinorPoleComponents

__all__ = [
    "MaxwellCurlField",
    "RamanujanCrystal",
    "HopfSpinorField",
    "SpinorPoleComponents",
]
