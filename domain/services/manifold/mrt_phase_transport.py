"""Dissipative Parallel Transport Service on Hopf U(1) Principal Fiber Bundle (ZENIN v2.4+).

Neutralizes phase divergence and high-frequency tick aliasing through
dissipative memory relaxation and canonical reduction to the S¹ circle [-π, π].
Conforms to ISO/IEC 25010 (Fault Tolerance & Numerical Precision).
"""

from __future__ import annotations

import math
import numpy as np

_TWO_PI = 2.0 * math.pi


def reduce_angle_canonical(angle: float) -> float:
    """Project real angle into canonical interval [-π, π) without transcendental drift."""
    if not np.isfinite(angle):
        return 0.0
    return ((float(angle) + math.pi) % _TWO_PI) - math.pi


def transport_phase_step(
    div_e: float,
    curl_b_norm: float,
    prev_phase_delta: float,
    delta_time: float,
    gamma_dissipation: float = 0.15,
) -> tuple[float, float, float]:
    """Propagate relative phase difference Δθ with Ornstein-Uhlenbeck dissipative relaxation.

    Governing ODE:
        d(Δθ)/dt = div(E) + ‖∇ × B‖ - γ_θ Δθ

    Discretized step:
        Δθ_{t} = (Δθ_{t-1} + (div_e + ‖∇×B‖) * Δt) * exp(-γ * Δt)
        followed by canonical modulo-2π reduction.

    Args:
        div_e: Divergence of 3D observable field (Rosa Roja E-field).
        curl_b_norm: Euclidean norm of MRT magnetic vorticity ‖∇ × B‖.
        prev_phase_delta: Previous relative phase Δθ_{t-1} ∈ [-π, π].
        delta_time: Time delta Δt > 0 in seconds.
        gamma_dissipation: Dissipative decay rate γ_θ > 0 (prevents unbounded phase memory).

    Returns:
        tuple (delta_theta, cos_delta_theta, sin_delta_theta) on [-π, π].
    """
    dt = max(1e-6, float(delta_time))
    gamma = max(1e-4, float(gamma_dissipation))
    decay = math.exp(-gamma * dt)

    d_e = float(div_e) if np.isfinite(div_e) else 0.0
    c_b = max(0.0, float(curl_b_norm)) if np.isfinite(curl_b_norm) else 0.0
    prev_theta = reduce_angle_canonical(prev_phase_delta)

    # Integrated flux step with dissipative contraction
    raw_next_theta = (prev_theta + (d_e + c_b) * dt) * decay
    next_theta = reduce_angle_canonical(raw_next_theta)

    cos_theta = math.cos(next_theta)
    sin_theta = math.sin(next_theta)

    return next_theta, cos_theta, sin_theta
