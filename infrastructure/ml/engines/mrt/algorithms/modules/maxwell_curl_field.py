"""Maxwell Circulation and Vorticity Field Module for MRT Engine (ZENIN v2.4+).

Dual-Engine Architecture (Conjugate Mirror):
    - Rosa Roja tracks observable displacement field (Maxwell E).
    - MaxwellCurlField evaluates rotational magnetic induction (Maxwell B) and
      vorticity ‖∇ × B‖ from the antisymmetric spin tensor Ω = 0.5(J - Jᵀ).
"""

from __future__ import annotations

import math
from typing import Sequence, cast
import numpy as np


class MaxwellCurlField:
    """Computes differential vorticity and phase-space circulation in ℝ³ and ℝᵈ."""

    def __init__(self, epsilon_stability: float = 1e-6) -> None:
        self._eps = max(1e-9, float(epsilon_stability))

    def extract_vorticity_tensor(self, jacobian: np.ndarray) -> np.ndarray:
        """Extract antisymmetric spin/vorticity tensor Ω = 0.5 * (J - Jᵀ)."""
        j_arr = np.asarray(jacobian, dtype=np.float64)
        if j_arr.ndim != 2 or j_arr.shape[0] != j_arr.shape[1]:
            raise ValueError(f"Expected square Jacobian, received {j_arr.shape}")
        return cast(np.ndarray, 0.5 * (j_arr - j_arr.T))

    def compute_curl_norm(self, jacobian: np.ndarray) -> float:
        """Compute Frobenius norm of rotational circulation ‖∇ × B‖ = √2 ‖Ω‖_F."""
        omega = self.extract_vorticity_tensor(jacobian)
        return float(math.sqrt(2.0) * float(np.linalg.norm(omega, ord="fro")))

    def compute_curl_vector_3d(self, jacobian: np.ndarray) -> np.ndarray:
        """Calculate exact 3D curl vector: (∂F_z/∂y - ∂F_y/∂z, ∂F_x/∂z - ∂F_z/∂x, ∂F_y/∂x - ∂F_x/∂y)."""
        j_arr = np.asarray(jacobian, dtype=np.float64)
        if j_arr.shape != (3, 3):
            return np.zeros(3, dtype=np.float64)
        wx = j_arr[2, 1] - j_arr[1, 2]
        wy = j_arr[0, 2] - j_arr[2, 0]
        wz = j_arr[1, 0] - j_arr[0, 1]
        return np.array([wx, wy, wz], dtype=np.float64)

    def compute_circulation_from_series(
        self,
        values: Sequence[float],
        delta_time: float = 1.0,
    ) -> tuple[float, float, float]:
        """Compute angular momentum vorticity in phase-space coordinates (x, v, a).

        Returns:
            tuple: (curl_norm, velocity, acceleration)
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 4:
            return 0.0, 0.0, 0.0

        dt = max(1e-4, float(delta_time))
        v1 = (float(arr[-1]) - float(arr[-2])) / dt
        v0 = (float(arr[-2]) - float(arr[-3])) / dt
        vm1 = (float(arr[-3]) - float(arr[-4])) / dt

        a1 = (v1 - v0) / dt
        a0 = (v0 - vm1) / dt
        jerk = (a1 - a0) / dt

        # Areal angular velocity L = v × a in 3D derivative space
        lx = (a1 * jerk) - (v1 * a0)
        ly = (v1 * jerk) - (a1 * v0)
        lz = (v1 * a1) - (v0 * a0)

        curl_mag = float(math.sqrt(lx**2 + ly**2 + lz**2) / (abs(v1) + self._eps))
        return curl_mag, v1, a1
