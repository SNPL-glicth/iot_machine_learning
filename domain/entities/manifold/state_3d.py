"""Continuous 3D State Manifold Value Object for ZENIN Stochastic Engine.

Conforms to:
- ISO/IEC 25010: Reliability and Numerical Fault Tolerance.
- ISO/IEC 22989: Artificial Intelligence Continuous State Auditability.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .manifold_parameters import ManifoldBoundaryLimits

_DEFAULT_LIMITS = ManifoldBoundaryLimits()


@dataclass(frozen=True, slots=True)
class ManifoldState3D:
    """Continuous three-dimensional state vector on Riemannian Manifold M ⊂ ℝ³.

    Coordinates:
        mahalanobis_d (x₁): Metric distance / spatial deformation d_M ∈ [0, ∞).
        kuramoto_r    (x₂): Topological Kuramoto phase resonance order parameter r ∈ [0, 1].
        bayesian_p    (x₃): Epistemic belief / posterior confidence P ∈ [0, 1].
    """

    mahalanobis_d: float
    kuramoto_r: float
    bayesian_p: float

    def __post_init__(self) -> None:
        """Enforce domain invariants and numerical validity at boundary."""
        lim = _DEFAULT_LIMITS
        if not np.isfinite(self.mahalanobis_d) or self.mahalanobis_d < 0.0:
            val = float(np.nan_to_num(self.mahalanobis_d, nan=0.0, posinf=lim.max_mahalanobis_clip, neginf=0.0))
            object.__setattr__(self, "mahalanobis_d", max(0.0, val))

        if not np.isfinite(self.kuramoto_r):
            object.__setattr__(self, "kuramoto_r", 0.0)
        else:
            object.__setattr__(self, "kuramoto_r", max(0.0, min(1.0, float(self.kuramoto_r))))

        if not np.isfinite(self.bayesian_p):
            object.__setattr__(self, "bayesian_p", 0.0)
        else:
            object.__setattr__(self, "bayesian_p", max(0.0, min(1.0, float(self.bayesian_p))))

    def to_numpy(self) -> np.ndarray:
        """Export state as contiguous IEEE 754 float64 vector [x₁, x₂, x₃]ᵀ."""
        return np.array([self.mahalanobis_d, self.kuramoto_r, self.bayesian_p], dtype=np.float64)

    @classmethod
    def from_numpy(cls, vector: np.ndarray) -> ManifoldState3D:
        """Construct ManifoldState3D from a 3-element numerical sequence."""
        arr = np.asarray(vector, dtype=np.float64).flatten()
        if arr.size < 3:
            raise ValueError(f"State vector requires at least 3 components, received {arr.size}")
        return cls(
            mahalanobis_d=float(arr[0]),
            kuramoto_r=float(arr[1]),
            bayesian_p=float(arr[2]),
        )

    @property
    def is_synchronized(self) -> bool:
        """Phase-space coherence criterion (Kuramoto order parameter >= sync_threshold)."""
        return self.kuramoto_r >= _DEFAULT_LIMITS.kuramoto_sync_threshold

    @property
    def metric_tension(self) -> float:
        """Geometric deformation energy scale E = 0.5 * d_M²."""
        return 0.5 * (self.mahalanobis_d**2)

    def to_iso_audit_dict(self) -> dict[str, float]:
        """ISO/IEC 22989 telemetry serialization for transparent audit trails."""
        return {
            "x1_mahalanobis_d": self.mahalanobis_d,
            "x2_kuramoto_r": self.kuramoto_r,
            "x3_bayesian_p": self.bayesian_p,
            "metric_tension_energy": self.metric_tension,
            "is_phase_synchronized": 1.0 if self.is_synchronized else 0.0,
        }
