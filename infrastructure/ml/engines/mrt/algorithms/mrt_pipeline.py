"""MRT Internal Conjugate Processing Pipeline (-x) (ZENIN v2.4+).

Dual-Engine Architecture (Conjugate Mirror):
    - Rosa Roja executes Forward Kinematics (Ingestion → Rhythm → Attractor).
    - MRTPipeline executes Inverse Kinematics (Curl → Ramanujan 4D Sink → Rebound Wave).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from .modules.hopf_spinor_field import HopfSpinorField
from .modules.maxwell_curl_field import MaxwellCurlField
from .modules.ramanujan_crystal import RamanujanCrystal


@dataclass(frozen=True, slots=True)
class MRTExecutionResult:
    """Rich execution result from MRT Conjugate Core (mirror of RosaRojaResult)."""

    rebound_target: float
    rebound_trajectory: list[float]
    rebound_direction: str
    rebound_magnitude: float
    z2_amplitude: float
    theta2_phase: float
    vorticity_curl: float
    frob_deformation_rate: float
    ramanujan_trace: float
    dissipation_factor: float
    confidence: float
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)


class MRTPipeline:
    """Orchestrates Maxwell vorticity, Ramanujan crystal, and Hopf spinor generation."""

    def __init__(
        self,
        elasticity_rebound: float = 0.85,
        horizon_steps: int = 5,
        strain_epsilon: float = 1e-3,
        gamma_dissipation: float = 0.15,
    ) -> None:
        self._elasticity = max(0.05, float(elasticity_rebound))
        self._horizon = max(1, horizon_steps)
        self._gamma = max(1e-4, float(gamma_dissipation))
        self._curl_field = MaxwellCurlField()
        self._crystal = RamanujanCrystal()
        self._spinor_field = HopfSpinorField(strain_epsilon=strain_epsilon)
        self._prev_phase: float = 0.0
        self._prev_velocity: float = 0.0

    def reset(self) -> None:
        """Reset temporal phase memory and velocity registers."""
        self._prev_phase = 0.0
        self._prev_velocity = 0.0

    def process(
        self,
        values: Sequence[float],
        delta_time: float = 1.0,
        nominal_certainty: float = 1.0,
    ) -> MRTExecutionResult:
        """Execute full inverse kinematics conjugate cycle."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 4:
            fallback = float(arr[-1]) if arr.size > 0 else 0.0
            return MRTExecutionResult(
                rebound_target=fallback, rebound_trajectory=[fallback],
                rebound_direction="stable", rebound_magnitude=0.0,
                z2_amplitude=0.0, theta2_phase=0.0, vorticity_curl=0.0,
                frob_deformation_rate=0.0, ramanujan_trace=0.0,
                dissipation_factor=1.0, confidence=0.0, status="insufficient_history",
            )

        dt = max(1e-4, float(delta_time))

        # 1. Maxwell Circulation Vorticity & Phase-Space Embedding
        curl_b, v_curr, a_curr = self._curl_field.compute_circulation_from_series(values, dt)
        frob_rate = float(math.hypot(abs(v_curr - self._prev_velocity) / dt, abs(a_curr)))
        self._prev_velocity = v_curr

        # 2. Dissipative Phase Transport on U(1) with time-reversal (-i)
        decay = math.exp(-self._gamma * dt)
        raw_phase = (self._prev_phase - (curl_b + frob_rate) * dt) * decay
        theta2 = ((raw_phase + math.pi) % (2.0 * math.pi)) - math.pi
        self._prev_phase = theta2

        # 3. Ramanujan 4D Symplectic Crystal Dissipation
        j_3d = np.diag([a_curr, v_curr, -abs(curl_b)])
        j_4d = self._crystal.build_augmented_4d_tensor(j_3d, frob_rate, np.array([arr[-1], v_curr, a_curr]))
        d_factor, tr_val = self._crystal.evaluate_crystal_dissipation(j_4d)

        # 4. Rational Hopf Spinor Negative Pole z2
        pole_z2 = self._spinor_field.evaluate_pole_z2(
            deformation_rate=frob_rate, phase_theta2=theta2,
            crystal_dissipation=d_factor, nominal_certainty=nominal_certainty,
        )

        # 5. Inverse Kinematics Multi-Step Rebound Trajectory (-x)
        rebound_gain = self._elasticity * (1.0 + (curl_b / (1.0 + curl_b)))
        v_rebound = -v_curr * rebound_gain
        x_last = float(arr[-1])
        rebound_traj: list[float] = []
        for step in range(1, self._horizon + 1):
            t_decay = math.exp(-self._gamma * step * dt)
            x_step = x_last + (v_rebound * step * dt * t_decay)
            rebound_traj.append(float(x_step))

        target = rebound_traj[0]
        direction = "up" if v_rebound > 1e-5 else ("down" if v_rebound < -1e-5 else "stable")
        mag = abs(v_rebound * dt)

        return MRTExecutionResult(
            rebound_target=target, rebound_trajectory=rebound_traj,
            rebound_direction=direction, rebound_magnitude=mag,
            z2_amplitude=pole_z2.z_amplitude, theta2_phase=theta2,
            vorticity_curl=curl_b, frob_deformation_rate=frob_rate,
            ramanujan_trace=tr_val, dissipation_factor=d_factor,
            confidence=float(max(0.0, min(1.0, pole_z2.z_amplitude))),
            status="ok", evidence={"v_curr": v_curr, "v_rebound": v_rebound, "z2_real": pole_z2.real_part},
        )
