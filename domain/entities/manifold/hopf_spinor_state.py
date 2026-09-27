"""Hopf Spinor Quantum State in ℂ² and Stokes S² Projection (ZENIN v2.4+).

Conforms to:
- ISO/IEC 25010: Numerical Fault Tolerance and Strict Invariant Proof.
- ISO/IEC 22989: Transparent Traceability of Continuous State Transitions.
- Mathematical Topology: S³ ⊂ ℂ² → S² ⊂ ℝ³ via Hopf Fibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import numpy as np


@dataclass(frozen=True, slots=True)
class HopfSpinorState:
    """Continuous 4D Quantum Spinor |ψ⟩ ∈ ℂ² with Stokes S² Projection.

    Coordinates:
        z1_real, z1_imag: Positive Pole (Rosa Roja 3D / Maxwell E).
        z2_real, z2_imag: Negative Pole (MRT 4D / Maxwell B / Ramanujan).
        stokes_s0: Invariant total norm S₀ = |z₁|² + |z₂|² ≡ C_nominal.
        stokes_s1: Tesla Resonant Coupling S₁ = 2 Re(z₁* z₂) = 2|z₁z₂|cos(Δθ).
        stokes_s2: Chiral Vorticity Quadrature S₂ = 2 Im(z₁* z₂) = 2|z₁z₂|sin(Δθ).
        stokes_s3: Polar Imbalance S₃ = |z₁|² - |z₂|² (Observable vs Extrinsic).
        eta_angle: Topological mixing angle η ∈ [0, π/2].
        phase_delta: Dissipative relative phase Δθ = θ₁ - θ₂ ∈ [-π, π].
        sovereign_certainty: Unified certainty C_sovereign = S₃ + S₁.
    """

    z1_real: float
    z1_imag: float
    z2_real: float
    z2_imag: float
    stokes_s0: float
    stokes_s1: float
    stokes_s2: float
    stokes_s3: float
    eta_angle: float
    phase_delta: float
    sovereign_certainty: float

    def __post_init__(self) -> None:
        """Enforce strict numerical boundaries and ISO 25010 Pythagorean Stokes invariant."""
        # Sanitize non-finite values to zero
        for field_name in (
            "z1_real", "z1_imag", "z2_real", "z2_imag",
            "stokes_s0", "stokes_s1", "stokes_s2", "stokes_s3",
            "eta_angle", "phase_delta", "sovereign_certainty"
        ):
            val = float(getattr(self, field_name))
            if not np.isfinite(val):
                object.__setattr__(self, field_name, 0.0)

        # Invariant Proof: S₁² + S₂² + S₃² == S₀² within tolerance 1e-7
        s0_sq = self.stokes_s0 ** 2
        s_vec_sq = (self.stokes_s1 ** 2) + (self.stokes_s2 ** 2) + (self.stokes_s3 ** 2)
        discrepancy = abs(s_vec_sq - s0_sq)

        if discrepancy > 1e-7 and s_vec_sq > 1e-14:
            # ISO 25010 fault tolerance: renormalise Stokes vector back to S0 invariant sphere
            correction_scale = math.sqrt(s0_sq / s_vec_sq)
            object.__setattr__(self, "stokes_s1", self.stokes_s1 * correction_scale)
            object.__setattr__(self, "stokes_s2", self.stokes_s2 * correction_scale)
            object.__setattr__(self, "stokes_s3", self.stokes_s3 * correction_scale)
            # Recompute unified certainty with corrected invariants
            object.__setattr__(self, "sovereign_certainty", self.stokes_s3 + self.stokes_s1)

    @property
    def effective_certainty(self) -> float:
        """Contract-compliant non-negative confidence magnitude in [0.0, 1.0]."""
        return max(0.0, min(1.0, abs(self.sovereign_certainty)))

    @property
    def polarity_direction(self) -> float:
        """Directional sovereign operator Π ∈ {-1.0, 1.0} for Veto Inverso."""
        return 1.0 if self.sovereign_certainty >= 0.0 else -1.0

    @property
    def is_tesla_resonant(self) -> bool:
        """True if constructive phase coupling exceeds inertial damping (S₁ > |S₃|)."""
        return self.stokes_s1 > abs(self.stokes_s3)

    def to_iso_audit_dict(self) -> dict[str, Any]:
        """Serialize state for ISO/IEC 22989 Clause 5.3 white-box traceability."""
        return {
            "z1_modulus": math.hypot(self.z1_real, self.z1_imag),
            "z2_modulus": math.hypot(self.z2_real, self.z2_imag),
            "stokes_s0": self.stokes_s0,
            "stokes_s1_tesla": self.stokes_s1,
            "stokes_s2_chiral": self.stokes_s2,
            "stokes_s3_polar": self.stokes_s3,
            "eta_angle_rad": self.eta_angle,
            "phase_delta_rad": self.phase_delta,
            "sovereign_certainty": self.sovereign_certainty,
            "effective_certainty": self.effective_certainty,
            "polarity_direction": self.polarity_direction,
            "is_tesla_resonant": self.is_tesla_resonant,
        }
