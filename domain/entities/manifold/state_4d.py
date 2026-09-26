"""Extrinsic 4D State Manifold Value Object for ZENIN Stochastic Engine.

Conforms to:
- ISO/IEC 25010: Fault Tolerance and Bounded Mathematical Recovery.
- ISO/IEC 22989: Traceability of Non-linear AI State Elevation and Anomaly Jumps.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .manifold_parameters import ManifoldBoundaryLimits
from .state_3d import ManifoldState3D

_DEFAULT_LIMITS = ManifoldBoundaryLimits()


@dataclass(frozen=True, slots=True)
class ManifoldState4D:
    """Extrinsic four-dimensional state vector on augmented manifold M₄ ⊂ ℝ⁴.

    Coordinates:
        mahalanobis_d       (x₁): Metric distance / spatial deformation d_M ∈ [0, ∞).
        kuramoto_r          (x₂): Kuramoto phase resonance order parameter r ∈ [0, 1].
        bayesian_p          (x₃): Epistemic belief / posterior confidence P ∈ [0, 1].
        extrinsic_rate_x4   (x₄): Rate of change of the geometric metric tensor J̇.
        frobenius_norm_j_dot: Raw Frobenius norm ||dJ/dt||_F indicating deformation velocity.
        trigger_reason      : Structured diagnostic identifier for the 4D elevation jump.
    """

    mahalanobis_d: float
    kuramoto_r: float
    bayesian_p: float
    extrinsic_rate_x4: float
    frobenius_norm_j_dot: float
    trigger_reason: str = "singularity_regularization"

    def __post_init__(self) -> None:
        """Enforce domain invariants and numerical validity under extreme volatility."""
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

        if not np.isfinite(self.extrinsic_rate_x4):
            object.__setattr__(
                self, "extrinsic_rate_x4", float(np.nan_to_num(self.extrinsic_rate_x4, nan=0.0))
            )

        if not np.isfinite(self.frobenius_norm_j_dot) or self.frobenius_norm_j_dot < 0.0:
            object.__setattr__(
                self, "frobenius_norm_j_dot", float(np.nan_to_num(self.frobenius_norm_j_dot, nan=0.0))
            )
            object.__setattr__(self, "frobenius_norm_j_dot", max(0.0, self.frobenius_norm_j_dot))

    def to_numpy(self) -> np.ndarray:
        """Export state as contiguous IEEE 754 float64 vector [x₁, x₂, x₃, x₄]ᵀ."""
        return np.array(
            [self.mahalanobis_d, self.kuramoto_r, self.bayesian_p, self.extrinsic_rate_x4],
            dtype=np.float64,
        )

    def extract_3d_projection(self) -> ManifoldState3D:
        """Project 4D state back to base 3D manifold coordinates."""
        return ManifoldState3D(
            mahalanobis_d=self.mahalanobis_d,
            kuramoto_r=self.kuramoto_r,
            bayesian_p=self.bayesian_p,
        )

    @classmethod
    def from_state_3d(
        cls,
        state_3d: ManifoldState3D,
        extrinsic_rate_x4: float,
        frobenius_norm_j_dot: float,
        trigger_reason: str = "divergence_instability",
    ) -> ManifoldState4D:
        """Construct 4D elevated state from a base 3D state and metric curvature rate."""
        return cls(
            mahalanobis_d=state_3d.mahalanobis_d,
            kuramoto_r=state_3d.kuramoto_r,
            bayesian_p=state_3d.bayesian_p,
            extrinsic_rate_x4=float(extrinsic_rate_x4),
            frobenius_norm_j_dot=float(frobenius_norm_j_dot),
            trigger_reason=str(trigger_reason),
        )

    @property
    def is_violent_shock(self) -> bool:
        """Metric acceleration flag (Frobenius rate of change > threshold)."""
        return self.frobenius_norm_j_dot > _DEFAULT_LIMITS.frobenius_shock_threshold

    def to_iso_audit_dict(self) -> dict[str, float | str]:
        """ISO/IEC 22989 telemetry serialization for 4D regularized state."""
        return {
            "x1_mahalanobis_d": self.mahalanobis_d,
            "x2_kuramoto_r": self.kuramoto_r,
            "x3_bayesian_p": self.bayesian_p,
            "x4_extrinsic_rate": self.extrinsic_rate_x4,
            "j_dot_frobenius_norm": self.frobenius_norm_j_dot,
            "is_violent_shock": 1.0 if self.is_violent_shock else 0.0,
            "trigger_reason": self.trigger_reason,
        }
