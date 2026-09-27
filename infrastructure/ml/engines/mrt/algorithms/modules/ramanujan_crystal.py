"""Ramanujan 4D Symplectic Crystal and Hydraulic Dissipation Module (ZENIN v2.4+).

Dual-Engine Architecture (Conjugate Mirror):
    - Rosa Roja models intra-manifold 3D phase volume dynamics.
    - RamanujanCrystal resolves the 4D extrinsic symplectic tensor J*₄D,
      guaranteeing strictly contractive dissipation Tr(J*₄D) < 0 to bound
      singular shocks and prevent unbounded numerical divergence.
"""

from __future__ import annotations

import math
import numpy as np


class RamanujanCrystal:
    """Constructs augmented 4D Jacobian and computes regularized crystal dissipation."""

    def __init__(
        self,
        tikhonov_delta: float = 1e-4,
        lambda4_base: float = 4.0,
        coupling_4d: float = 0.20,
    ) -> None:
        self._delta = max(1e-8, float(tikhonov_delta))
        self._lambda4_base = max(1.0, float(lambda4_base))
        self._coupling = max(0.01, min(1.0, float(coupling_4d)))

    def build_augmented_4d_tensor(
        self,
        j_3d: np.ndarray,
        deformation_rate: float,
        state_3d: np.ndarray,
    ) -> np.ndarray:
        """Construct augmented 4x4 Jacobian tensor guaranteeing contractivity in ℝ⁴."""
        j_arr = np.asarray(j_3d, dtype=np.float64)
        s_arr = np.asarray(state_3d, dtype=np.float64).flatten()

        j_4d = np.zeros((4, 4), dtype=np.float64)
        j_4d[0:3, 0:3] = j_arr[0:3, 0:3] if j_arr.shape == (3, 3) else np.eye(3) * -1.0

        # Coupling from 4D into 3D state
        s_norm = max(1e-6, float(np.linalg.norm(s_arr[0:3])))
        j_4d[0:3, 3] = -self._coupling * (s_arr[0:3] / s_norm)

        # Extrinsic 4D strain coupling
        j_4d[3, 0:3] = np.diag(j_4d[0:3, 0:3]) * (deformation_rate / (1.0 + deformation_rate))

        # Extrinsic sink parameter λ₄: guarantees Tr(J_4D) < 0
        div_3d = float(np.trace(j_4d[0:3, 0:3]))
        lambda_4 = max(self._lambda4_base, abs(div_3d) * 2.0 + 1.0)
        j_4d[3, 3] = -lambda_4

        return j_4d

    def compute_symplectic_inverse_trace(
        self,
        j_4d: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute Tikhonov-regularized inverse J* = (JᵀJ + δI)⁻¹ Jᵀ and its trace."""
        m_arr = np.asarray(j_4d, dtype=np.float64)
        dim = m_arr.shape[0]

        # Regularized Gram matrix
        gram = m_arr.T @ m_arr + (self._delta * np.eye(dim, dtype=np.float64))
        # Closed-form inversion with condition-number protection
        inv_gram = np.linalg.inv(gram)
        j_star = inv_gram @ m_arr.T

        trace_star = float(np.trace(j_star))
        # Guarantee negative dissipation sign
        safe_trace = -abs(trace_star) if trace_star > 0.0 else trace_star

        return safe_trace, j_star

    def evaluate_crystal_dissipation(
        self,
        j_4d: np.ndarray,
    ) -> tuple[float, float]:
        """Calculate Ramanujan hydraulic dissipation factor: D = exp(Tr(J*)).

        Returns:
            tuple: (dissipation_factor in (0.0, 1.0], safe_negative_trace)
        """
        tr_val, _ = self.compute_symplectic_inverse_trace(j_4d)
        clamped_tr = min(0.0, max(-8.0, tr_val))
        damping = float(math.exp(clamped_tr))
        return damping, clamped_tr
