"""Empirical Jacobian Estimation Domain Service.

Conforms to:
- ISO/IEC 25012:2008: Data quality, accuracy, credibility in time-series estimation.
- ISO/IEC 25010:2023: Fault tolerance, performance efficiency.
- Strict constraint: Line count <= 180 lines, pure domain service.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple
import numpy as np

from domain.entities.manifold.state_3d import ManifoldState3D
from .jacobian_tensor import compute_jacobian_tensor


class EmpiricalJacobianService:
    """Estimates local differential Jacobian tensor J = ∇F directly from observations."""

    def __init__(
        self,
        window_size: int = 15,
        tikhonov_regularization: float = 1e-4,
        max_condition_number: float = 1e4,
        min_samples: int = 5,
    ) -> None:
        """Initialize estimator with window size, regularization, and condition threshold."""
        self._window_size = max(5, window_size)
        self._delta_reg = max(1e-8, tikhonov_regularization)
        self._max_cond = max(10.0, max_condition_number)
        self._min_samples = max(4, min_samples)

    def estimate_empirical_jacobian(
        self,
        states: np.ndarray,
        time_deltas: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Estimate 3x3 Jacobian J = ∇F via Tikhonov-regularized multivariate Ridge regression.

        Solves local linear flow: Δx_{k+1} ≈ J · (x_k - x̄)
        Returns:
            tuple: (J_empirical in ℝ³ˣ³, condition_number, confidence_weight in [0, 1])
        """
        arr = np.asarray(states, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < self._min_samples:
            return np.zeros((3, 3), dtype=np.float64), float("inf"), 0.0

        w_samples = min(arr.shape[0], self._window_size)
        x_win = arr[-w_samples:]

        # Differences: velocities or discrete transitions Δx
        if time_deltas is not None and len(time_deltas) >= (w_samples - 1):
            dt = np.asarray(time_deltas[-(w_samples - 1):], dtype=np.float64).reshape(-1, 1)
            dt = np.maximum(1e-4, dt)
            dy = (x_win[1:] - x_win[:-1]) / dt
        else:
            dy = x_win[1:] - x_win[:-1]

        # Centered design matrix X_c in ℝ^(N x 3)
        xc = x_win[:-1] - np.mean(x_win[:-1], axis=0, keepdims=True)

        # Gram matrix G = X_cᵀ X_c ∈ ℝ³ˣ³
        gram = xc.T @ xc
        gram_reg = gram + (self._delta_reg * np.eye(3, dtype=np.float64))

        # Check conditioning of the empirical excitation
        try:
            s_vals = np.linalg.svd(gram_reg, compute_uv=False)
            cond = float(s_vals[0] / max(1e-12, s_vals[-1]))
        except Exception:
            cond = float("inf")

        if math.isinf(cond) or math.isnan(cond) or cond > self._max_cond:
            # Collinear or unexcited subspace: empirical weight collapses to 0
            return np.zeros((3, 3), dtype=np.float64), cond, 0.0

        # Exact closed-form Ridge solution in ℝ³ˣ³: J = (Yᵀ X_c) (X_cᵀ X_c + δI)⁻¹
        try:
            inv_gram = np.linalg.inv(gram_reg)
            j_emp = (dy.T @ xc) @ inv_gram
        except Exception:
            return np.zeros((3, 3), dtype=np.float64), cond, 0.0

        # Confidence weight: high when well-conditioned and populated
        sample_ratio = min(1.0, float(w_samples) / float(self._window_size))
        cond_ratio = max(0.0, 1.0 - (math.log10(max(1.0, cond)) / math.log10(self._max_cond)))
        confidence = float(max(0.0, min(1.0, sample_ratio * cond_ratio)))

        return j_emp, cond, confidence

    def blend_jacobian(
        self,
        current_state: ManifoldState3D | np.ndarray,
        history_states: np.ndarray,
        beta_max: float = 0.80,
        time_deltas: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Blend analytical canonical Jacobian with empirical observable flow.

        Equation:
            J_blended = (1 - β) · J_analytical + β · J_empirical
            where β = β_max · confidence_empirical ∈ [0, β_max].

        Returns:
            tuple: (J_blended, divergence = Tr(J_blended), beta_applied)
        """
        # 1. Base analytical Jacobian (canonical prior)
        j_ana = compute_jacobian_tensor(current_state)

        # 2. Local empirical Jacobian
        j_emp, _, conf = self.estimate_empirical_jacobian(history_states, time_deltas=time_deltas)

        # 3. Soft blending protecting stability
        beta = float(max(0.0, min(float(beta_max), float(beta_max) * conf)))
        j_blended = ((1.0 - beta) * j_ana) + (beta * j_emp)

        divergence = float(np.trace(j_blended))
        return j_blended, divergence, beta
