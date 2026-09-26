"""Analytical Geometric Matrix (Jacobian Tensor) Engine on Manifold M ⊂ ℝ³.

Computes closed-form differential coordinate tensor J(x) = ∇F ∈ ℝ³ˣ³
coordinating metric deformation, topological phase-locking, and Bayesian belief.
"""

from __future__ import annotations

import numpy as np

from domain.entities.manifold.state_3d import ManifoldState3D
from .vector_field import VectorFieldConfig


def compute_jacobian_tensor(
    state: ManifoldState3D | np.ndarray,
    config: VectorFieldConfig | None = None,
) -> np.ndarray:
    """Evaluate exact analytical 3x3 Geometric Matrix J(x) = ∇F."""
    cfg = config or VectorFieldConfig()
    if hasattr(state, "mahalanobis_d") and hasattr(state, "kuramoto_r") and hasattr(state, "bayesian_p"):
        x1, x2, x3 = float(state.mahalanobis_d), float(state.kuramoto_r), float(state.bayesian_p)
    else:
        arr = np.asarray(state, dtype=np.float64).flatten()
        x1, x2, x3 = float(arr[0]), float(arr[1]), float(arr[2])

    J = np.zeros((3, 3), dtype=np.float64)

    # Row 1: ∂F1 / ∂x
    J[0, 0] = -cfg.lambda_1 + cfg.alpha_12 * (1.0 - x2) + cfg.alpha_13 * (1.0 - x3)
    J[0, 1] = -cfg.alpha_12 * x1
    J[0, 2] = -cfg.alpha_13 * x1

    # Row 2: ∂F2 / ∂x
    x1_sq = x1 * x1
    denom_sq = (1.0 + x1_sq) ** 2
    d_mahal_damping = (2.0 * x1) / denom_sq
    J[1, 0] = -cfg.beta_21 * d_mahal_damping * x2
    J[1, 1] = cfg.gamma_2 * (1.0 - 2.0 * x2) - cfg.beta_21 * (x1_sq / (1.0 + x1_sq)) - cfg.beta_23 * x3
    J[1, 2] = cfg.beta_23 * (1.0 - x2)

    # Row 3: ∂F3 / ∂x
    clip_lim = cfg.limits.sech_input_clip
    sech2_x1 = 1.0 / (np.cosh(np.clip(x1, -clip_lim, clip_lim)) ** 2)
    J[2, 0] = -cfg.eta_31 * sech2_x1 * x3
    J[2, 1] = cfg.eta_32 * (1.0 - x3)
    J[2, 2] = -cfg.mu_3 - cfg.eta_31 * np.tanh(x1) - cfg.eta_32 * x2

    return J


def compute_jacobian_batch(
    states: np.ndarray,
    config: VectorFieldConfig | None = None,
) -> np.ndarray:
    """Batch evaluation of Jacobian tensors without Python loops."""
    cfg = config or VectorFieldConfig()
    arr = np.asarray(states, dtype=np.float64)
    b = arr.shape[0]
    x1 = arr[:, 0]
    x2 = arr[:, 1]
    x3 = arr[:, 2]

    J = np.zeros((b, 3, 3), dtype=np.float64)

    # Vectorized Row 1
    J[:, 0, 0] = -cfg.lambda_1 + cfg.alpha_12 * (1.0 - x2) + cfg.alpha_13 * (1.0 - x3)
    J[:, 0, 1] = -cfg.alpha_12 * x1
    J[:, 0, 2] = -cfg.alpha_13 * x1

    # Vectorized Row 2
    x1_sq = x1 * x1
    denom_sq = (1.0 + x1_sq) ** 2
    d_mahal = (2.0 * x1) / denom_sq
    J[:, 1, 0] = -cfg.beta_21 * d_mahal * x2
    J[:, 1, 1] = cfg.gamma_2 * (1.0 - 2.0 * x2) - cfg.beta_21 * (x1_sq / (1.0 + x1_sq)) - cfg.beta_23 * x3
    J[:, 1, 2] = cfg.beta_23 * (1.0 - x2)

    # Vectorized Row 3
    clip_lim = cfg.limits.sech_input_clip
    sech2 = 1.0 / (np.cosh(np.clip(x1, -clip_lim, clip_lim)) ** 2)
    J[:, 2, 0] = -cfg.eta_31 * sech2 * x3
    J[:, 2, 1] = cfg.eta_32 * (1.0 - x3)
    J[:, 2, 2] = -cfg.mu_3 - cfg.eta_31 * np.tanh(x1) - cfg.eta_32 * x2

    return J


def verify_jacobian_numerical(
    state: np.ndarray,
    eps: float = 1e-7,
    config: VectorFieldConfig | None = None,
) -> float:
    """Verify analytical Jacobian against numerical central differences."""
    from .vector_field import compute_vector_field

    J_ana = compute_jacobian_tensor(state, config)
    J_num = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        dx = np.zeros(3, dtype=np.float64)
        dx[i] = eps
        fp = compute_vector_field(state + dx, config)
        fm = compute_vector_field(state - dx, config)
        J_num[:, i] = (fp - fm) / (2.0 * eps)
    return float(np.max(np.abs(J_ana - J_num)))
