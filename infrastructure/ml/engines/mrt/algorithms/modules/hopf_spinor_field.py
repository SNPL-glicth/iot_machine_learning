"""Rational Algebraic Hopf Spinor Field Module for MRT Engine (ZENIN v2.4+).

Dual-Engine Architecture (Conjugate Mirror):
    - Evaluates the negative quantum pole z₂ ∈ ℂ and projects onto Stokes S² manifold.
    - Zero Transcendental Overhead: computes exact half-angle amplitudes using
      rational square roots, preserving sub-microsecond HFT latency budgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class SpinorPoleComponents:
    """Wave amplitude, phase coordinates, and Stokes projections for a quantum pole."""

    z_amplitude: float
    theta_phase: float
    real_part: float
    imag_part: float
    energy_norm_sq: float


class HopfSpinorField:
    """Evaluates 4D conjugate spinor amplitudes and Stokes parameters via rational algebra."""

    def __init__(self, strain_epsilon: float = 1e-3) -> None:
        self._eps = max(1e-7, float(strain_epsilon))

    def compute_rational_amplitudes(
        self,
        deformation_rate: float,
        nominal_certainty: float = 1.0,
    ) -> tuple[float, float, float]:
        """Compute exact Hopf amplitudes (a1, a2) without transcendental functions.

        Derivation:
            u = ‖J̇‖_F / ε
            cos(η) = 1 / sqrt(1 + u²)
            cos(η/2) = sqrt((1 + cos(η)) / 2)
            sin(η/2) = sqrt(max(0, (1 - cos(η)) / 2))

        Returns:
            tuple: (a1_positive_3d, a2_negative_4d, cos_eta)
        """
        c_nom = max(0.0, min(1.0, float(nominal_certainty)))
        u = max(0.0, float(deformation_rate)) / self._eps

        hypot_u = math.hypot(1.0, u)
        cos_eta = 1.0 / hypot_u

        cos_half = math.sqrt(0.5 * (1.0 + cos_eta))
        sin_half = math.sqrt(max(0.0, 0.5 * (1.0 - cos_eta)))

        sqrt_c = math.sqrt(c_nom)
        return sqrt_c * cos_half, sqrt_c * sin_half, cos_eta

    def evaluate_pole_z2(
        self,
        deformation_rate: float,
        phase_theta2: float,
        crystal_dissipation: float = 1.0,
        nominal_certainty: float = 1.0,
    ) -> SpinorPoleComponents:
        """Construct the MRT Negative Pole z₂ ∈ ℂ incorporating Ramanujan crystal damping.

        Formula:
            z₂ = √(C_nominal) * sin(η/2) * exp(-i θ₂) * D_Ramanujan
        """
        _, a2_raw, _ = self.compute_rational_amplitudes(deformation_rate, nominal_certainty)
        damp = max(0.0, min(1.0, float(crystal_dissipation)))
        z2_amp = a2_raw * damp

        theta = float(phase_theta2)
        # Time-reversed conjugate wave exp(-i θ₂)
        real_p = z2_amp * math.cos(theta)
        imag_p = -z2_amp * math.sin(theta)

        return SpinorPoleComponents(
            z_amplitude=z2_amp,
            theta_phase=theta,
            real_part=real_p,
            imag_part=imag_p,
            energy_norm_sq=z2_amp ** 2,
        )

    def evaluate_stokes_sovereign(
        self,
        z1_amp: float,
        z2_amp: float,
        phase_delta: float,
    ) -> tuple[float, float, float, float]:
        """Compute Stokes invariants (S₀, S₁, S₃) and Sovereign Certainty C_Sovereign = S₃ + S₁.

        Returns:
            tuple: (sovereign_certainty, stokes_s0, stokes_s1, stokes_s3)
        """
        a1, a2 = max(0.0, float(z1_amp)), max(0.0, float(z2_amp))
        s0 = (a1 ** 2) + (a2 ** 2)
        s3 = (a1 ** 2) - (a2 ** 2)
        s1 = 2.0 * a1 * a2 * math.cos(float(phase_delta))
        c_sovereign = s3 + s1
        return c_sovereign, s0, s1, s3
