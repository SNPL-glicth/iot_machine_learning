"""Continuous 3D Vector Field Flow Engine on Riemannian Manifold M ⊂ ℝ³.

Computes F(x) = dx/dt coupling Mahalanobis metric, Kuramoto phase synchrony,
and Bayesian belief dynamics under strict hexagonal domain isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from domain.entities.manifold.manifold_parameters import ManifoldBoundaryLimits
from domain.entities.manifold.state_3d import ManifoldState3D


@dataclass(frozen=True, slots=True)
class VectorFieldConfig:
    """Hyperparameters and cross-coupling coefficients for the 3D manifold flow."""

    lambda_1: float = 1.0  # Mahalanobis relaxation rate
    alpha_12: float = 0.8  # Coupling from Kuramoto dephasing to Mahalanobis
    alpha_13: float = 0.5  # Coupling from Bayesian uncertainty to Mahalanobis
    x1_star: float = 1.0   # Nominal Mahalanobis metric equilibrium

    gamma_2: float = 1.2   # Kuramoto intrinsic phase-locking rate
    beta_21: float = 1.5   # Shock dephasing sensitivity from Mahalanobis metric
    beta_23: float = 0.7   # Reinforcement from Bayesian confidence

    mu_3: float = 0.8      # Bayesian evidence relaxation rate
    eta_31: float = 0.6    # Posterior penalty under geometric dispersion
    eta_32: float = 0.5    # Resonance gain from topological phase coherence
    p_target: float = 0.8  # Nominal Bayesian equilibrium target

    limits: ManifoldBoundaryLimits = field(default_factory=ManifoldBoundaryLimits)


def compute_vector_field(
    state: ManifoldState3D | np.ndarray,
    config: VectorFieldConfig | None = None,
) -> np.ndarray:
    """Evaluate continuous velocity vector F(x) = dx/dt ∈ ℝ³.

    Args:
        state: State instance or 3-element float array [d_M, r, P]ᵀ.
        config: Flow parameters; defaults to calibrated standard values.

    Returns:
        np.ndarray of shape (3,) representing instantaneous rates of change.
    """
    cfg = config or VectorFieldConfig()
    if hasattr(state, "mahalanobis_d") and hasattr(state, "kuramoto_r") and hasattr(state, "bayesian_p"):
        x1, x2, x3 = float(state.mahalanobis_d), float(state.kuramoto_r), float(state.bayesian_p)
    else:
        arr = np.asarray(state, dtype=np.float64).flatten()
        x1, x2, x3 = float(arr[0]), float(arr[1]), float(arr[2])

    f1 = (
        -cfg.lambda_1 * (x1 - cfg.x1_star)
        + cfg.alpha_12 * (1.0 - x2) * x1
        + cfg.alpha_13 * (1.0 - x3) * x1
    )
    x1_sq = x1 * x1
    mahal_damping = x1_sq / (1.0 + x1_sq)
    f2 = (
        cfg.gamma_2 * x2 * (1.0 - x2)
        - cfg.beta_21 * mahal_damping * x2
        + cfg.beta_23 * x3 * (1.0 - x2)
    )
    f3 = (
        cfg.mu_3 * (cfg.p_target - x3)
        - cfg.eta_31 * np.tanh(x1) * x3
        + cfg.eta_32 * x2 * (1.0 - x3)
    )
    return np.array([f1, f2, f3], dtype=np.float64)


def compute_vector_field_batch(
    states: np.ndarray,
    config: VectorFieldConfig | None = None,
) -> np.ndarray:
    """Vectorized batch evaluation of F(X) for high-frequency streaming.

    Args:
        states: Matrix of shape (B, 3) where columns are [d_M, r, P].
        config: Flow parameters.

    Returns:
        Matrix of velocities of shape (B, 3) computed without Python loops.
    """
    cfg = config or VectorFieldConfig()
    arr = np.asarray(states, dtype=np.float64)
    x1 = arr[:, 0]
    x2 = arr[:, 1]
    x3 = arr[:, 2]

    f1 = (
        -cfg.lambda_1 * (x1 - cfg.x1_star)
        + cfg.alpha_12 * (1.0 - x2) * x1
        + cfg.alpha_13 * (1.0 - x3) * x1
    )
    x1_sq = x1 * x1
    mahal_damping = x1_sq / (1.0 + x1_sq)
    f2 = (
        cfg.gamma_2 * x2 * (1.0 - x2)
        - cfg.beta_21 * mahal_damping * x2
        + cfg.beta_23 * x3 * (1.0 - x2)
    )
    f3 = (
        cfg.mu_3 * (cfg.p_target - x3)
        - cfg.eta_31 * np.tanh(x1) * x3
        + cfg.eta_32 * x2 * (1.0 - x3)
    )
    return np.column_stack([f1, f2, f3])


def integrate_rk4_step(
    state: np.ndarray,
    delta_time: float,
    config: VectorFieldConfig | None = None,
) -> np.ndarray:
    """Non-linear 4th-order Runge-Kutta integrator for dx/dt = F(x)."""
    cfg = config or VectorFieldConfig()
    dt = float(delta_time)
    k1 = compute_vector_field(state, cfg)
    k2 = compute_vector_field(state + 0.5 * dt * k1, cfg)
    k3 = compute_vector_field(state + 0.5 * dt * k2, cfg)
    k4 = compute_vector_field(state + dt * k3, cfg)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    max_d = cfg.limits.max_mahalanobis_clip
    return np.clip(next_state, a_min=[0.0, 0.0, 0.0], a_max=[max_d, 1.0, 1.0])
