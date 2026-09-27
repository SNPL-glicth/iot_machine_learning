"""Phase Conjugator and Market Vorticity Filter for MRT Engine (ZENIN v2.4+).

Dual-Engine Architecture:
    - Rosa Roja (Observable Pole / 3D E-Field): Forward Kinematics, inertia & rhythm.
    - MRT Phase Conjugator (Dark Pole / 4D B-Field): Inverse Kinematics, vorticity ‖∇ × B‖
      and phase conjugation under Maxwell time-reversal symmetry (B → -B, p → -p).

Eliminates transcendental overhead via Rational Algebraic Shortcut and bounds
phase evolution to the canonical circle S¹ ∈ [-π, π].
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


class PhaseConjugator:
    """Extracts phase-space circulation, metric deformation, and dissipative phase.

    Acts as the conjugate counterpart to Rosa Roja's MahalanobisFilter.
    """

    def __init__(
        self,
        epsilon_strain: float = 1e-3,
        gamma_dissipation: float = 0.15,
        min_history: int = 4,
    ) -> None:
        self._eps = max(1e-6, float(epsilon_strain))
        self._gamma = max(1e-4, float(gamma_dissipation))
        self._min_history = max(3, min_history)
        self._prev_phase: float = 0.0
        self._prev_velocity: float = 0.0

    @property
    def current_phase(self) -> float:
        return self._prev_phase

    def reset(self) -> None:
        """Reset temporal phase register and velocity memory."""
        self._prev_phase = 0.0
        self._prev_velocity = 0.0

    def compute_phase_conjugation(
        self,
        values: Sequence[float],
        delta_time: float = 1.0,
    ) -> tuple[float, float, float, float, float]:
        """Compute vorticity, metric deformation, rational amplitude, and conjugate phase.

        Args:
            values: Historical sequence of price or signal observations.
            delta_time: Time delta Δt > 0 between consecutive ticks.

        Returns:
            tuple: (z2_amplitude, theta_conjugate, vorticity_norm, frob_rate, velocity)
        """
        arr = np.asarray(values, dtype=np.float64)
        n = arr.size
        if n < self._min_history:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        dt = max(1e-4, float(delta_time))

        # 1. Discrete derivatives for phase-space embedding (x, v, a)
        x_curr = float(arr[-1])
        x_prev1 = float(arr[-2])
        x_prev2 = float(arr[-3])
        x_prev3 = float(arr[-4]) if n >= 4 else (2.0 * x_prev2 - x_prev1)

        v_curr = (x_curr - x_prev1) / dt
        v_prev = (x_prev1 - x_prev2) / dt
        a_curr = (v_curr - v_prev) / dt
        a_prev = (v_prev - (x_prev2 - x_prev3) / dt) / dt
        jerk = (a_curr - a_prev) / dt

        # 2. Phase-space circulation vorticity ‖∇ × B‖ via angular momentum cross product
        # L = r × v in phase space coordinates (v, a, j)
        curl_x = (a_curr * jerk) - (v_curr * a_prev)
        curl_y = (v_curr * jerk) - (a_curr * v_prev)
        curl_z = (v_curr * a_curr) - (v_prev * a_prev)
        vorticity = float(math.sqrt(curl_x**2 + curl_y**2 + curl_z**2) / (abs(v_curr) + self._eps))

        # 3. Metric deformation velocity rate ‖J̇‖_F
        accel_shock = abs(a_curr - a_prev) / dt
        vel_shock = abs(v_curr - self._prev_velocity) / dt
        frob_rate = float(math.hypot(accel_shock, vel_shock))
        self._prev_velocity = v_curr

        # 4. Rational Algebraic Shortcut for mixing angle η and 4D amplitude z2
        # u = ‖J̇‖_F / ε, cos(η) = 1 / sqrt(1 + u²), sin(η/2) = sqrt((1 - cos(η)) / 2)
        u = frob_rate / self._eps
        hypot_u = math.hypot(1.0, u)
        cos_eta = 1.0 / hypot_u
        sin_half_eta = math.sqrt(max(0.0, 0.5 * (1.0 - cos_eta)))

        # 5. Dissipative Phase Transport on U(1) with time-reversal conjugation (-i)
        decay = math.exp(-self._gamma * dt)
        next_raw_phase = (self._prev_phase - (vorticity + frob_rate) * dt) * decay
        # Canonical modulo 2π reduction to [-π, π]
        theta_conj = ((next_raw_phase + math.pi) % (2.0 * math.pi)) - math.pi
        self._prev_phase = theta_conj

        return sin_half_eta, theta_conj, vorticity, frob_rate, v_curr
