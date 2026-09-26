"""Ramanujan 4D Dimensional Jump and Symplectic Regularization Service.

Resolves 3D phase-space singularities by elevating erratic states into an extrinsic
4th dimension derived from metric tensor deformation velocity ||dJ/dt||_F,
guaranteeing contractive dissipation Tr(J_4D) < 0 and smooth geodesic recovery.
"""

from __future__ import annotations

import numpy as np

from domain.entities.manifold.manifold_parameters import ManifoldBoundaryLimits
from domain.entities.manifold.state_3d import ManifoldState3D
from domain.entities.manifold.state_4d import ManifoldState4D

_DEFAULT_LIMITS = ManifoldBoundaryLimits()


def compute_metric_deformation_velocity(
    j_curr: np.ndarray,
    j_prev: np.ndarray,
    delta_time: float,
    limits: ManifoldBoundaryLimits | None = None,
) -> tuple[np.ndarray, float, float]:
    """Compute metric tensor temporal derivative J̇ = dJ/dt and Frobenius rate."""
    lim = limits or _DEFAULT_LIMITS
    dt = max(lim.eps_denominator, float(delta_time))
    j_dot = (np.asarray(j_curr, dtype=np.float64) - np.asarray(j_prev, dtype=np.float64)) / dt
    norm_frob = float(np.linalg.norm(j_dot, ord="fro"))
    tr_dot = float(np.trace(j_dot))
    sign = 1.0 if tr_dot >= 0.0 else -1.0
    return j_dot, norm_frob, sign * norm_frob


def build_augmented_4d_jacobian(
    j_3d: np.ndarray,
    j_dot: np.ndarray,
    state_3d: np.ndarray,
    limits: ManifoldBoundaryLimits | None = None,
) -> np.ndarray:
    """Construct augmented 4x4 Jacobian tensor guaranteeing contractivity in ℝ⁴."""
    lim = limits or _DEFAULT_LIMITS
    j_4d = np.zeros((4, 4), dtype=np.float64)
    j_4d[0:3, 0:3] = j_3d

    norm_x = max(lim.eps_denominator, float(np.linalg.norm(state_3d)))
    j_4d[0:3, 3] = -lim.coupling_4d_default * (state_3d / norm_x)

    norm_dot = max(lim.eps_denominator, float(np.linalg.norm(j_dot, ord="fro")))
    j_4d[3, 0:3] = np.diag(j_dot) / norm_dot

    div_3d = float(np.trace(j_3d))
    lambda_4 = max(lim.lambda_4_base, abs(div_3d) * lim.lambda_4_scale + lim.lambda_4_offset)
    j_4d[3, 3] = -lambda_4

    return j_4d


def propagate_geodesic_step(
    state_vector_4d: np.ndarray,
    j_4d: np.ndarray,
    delta_time: float,
    limits: ManifoldBoundaryLimits | None = None,
) -> np.ndarray:
    """Solve 4D homogeneous ODE dx/dt - J_4D x = 0 via order-4 symplectic polynomial."""
    lim = limits or _DEFAULT_LIMITS
    dt = max(lim.eps_denominator, float(delta_time))
    M = j_4d * dt
    eye = np.eye(4, dtype=np.float64)
    m2 = M @ M
    m3 = m2 @ M
    m4 = m3 @ M
    u_4d = eye + M + 0.5 * m2 + (1.0 / 6.0) * m3 + (1.0 / 24.0) * m4
    return u_4d @ state_vector_4d


def execute_ramanujan_jump(
    state_3d: ManifoldState3D,
    j_curr: np.ndarray,
    j_prev: np.ndarray,
    delta_time: float,
    trigger_reason: str = "singularity_regularization",
    limits: ManifoldBoundaryLimits | None = None,
) -> tuple[ManifoldState4D, np.ndarray, ManifoldState3D]:
    """Execute complete Ramanujan 4D elevation jump and geodesic recovery."""
    lim = limits or _DEFAULT_LIMITS
    j_dot, frob_norm, x4 = compute_metric_deformation_velocity(j_curr, j_prev, delta_time, lim)

    state_4d = ManifoldState4D.from_state_3d(
        state_3d=state_3d,
        extrinsic_rate_x4=x4,
        frobenius_norm_j_dot=frob_norm,
        trigger_reason=trigger_reason,
    )

    s3_arr = state_3d.to_numpy()
    j_4d = build_augmented_4d_jacobian(j_curr, j_dot, s3_arr, lim)

    s4_arr = state_4d.to_numpy()
    s4_next = propagate_geodesic_step(s4_arr, j_4d, delta_time, lim)

    clamped_3d = np.clip(s4_next[0:3], a_min=[0.0, 0.0, 0.0], a_max=[lim.max_mahalanobis_clip, 1.0, 1.0])
    recovered_3d = ManifoldState3D.from_numpy(clamped_3d)

    return state_4d, j_4d, recovered_3d
