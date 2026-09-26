"""ISO/IEC 22989 and ISO/IEC 25010 Manifold Audit Record for ZENIN Stochastic Engine.

Provides complete mathematical observability and non-linear state auditability:
- Traceability of 3D/4D Riemannian State Transitions.
- Verification of Liouville Phase Volume Invariants: Tr(J) = div(F).
- Auditable Diagnostics for AI Explanations and Critical Action Authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .manifold_parameters import ManifoldBoundaryLimits
from .state_3d import ManifoldState3D
from .state_4d import ManifoldState4D

_DEFAULT_LIMITS = ManifoldBoundaryLimits()


@dataclass(frozen=True, slots=True)
class ManifoldAuditRecord:
    """Comprehensive geometric and topological audit record for a single inference step.

    Conforms to:
        - ISO/IEC 22989 Clause 5.3 (AI System Transparency & Traceability).
        - ISO/IEC 25010 Reliability (Numerical Fault Tolerance & Invariant Proof).
    """

    timestamp: float
    state_3d: ManifoldState3D
    divergence: float
    determinant: float
    volume_rate: float
    max_real_eigenvalue: float
    is_contractive: bool
    is_4d_projected: bool
    stability_verdict: str
    state_4d: ManifoldState4D | None = None
    j_frobenius_norm: float = 0.0

    def __post_init__(self) -> None:
        """Enforce strict numerical boundaries and guard against NaNs/Infinities."""
        lim = _DEFAULT_LIMITS
        if not np.isfinite(self.timestamp):
            object.__setattr__(self, "timestamp", 0.0)

        if not np.isfinite(self.divergence):
            object.__setattr__(
                self,
                "divergence",
                float(np.nan_to_num(self.divergence, nan=0.0, posinf=lim.divergence_clip_max, neginf=lim.divergence_clip_min)),
            )

        if not np.isfinite(self.determinant):
            object.__setattr__(
                self, "determinant", float(np.nan_to_num(self.determinant, nan=0.0))
            )

        if not np.isfinite(self.volume_rate):
            object.__setattr__(self, "volume_rate", self.divergence)

        if not np.isfinite(self.max_real_eigenvalue):
            object.__setattr__(
                self, "max_real_eigenvalue", float(np.nan_to_num(self.max_real_eigenvalue, nan=0.0))
            )

        if not np.isfinite(self.j_frobenius_norm) or self.j_frobenius_norm < 0.0:
            val = float(np.nan_to_num(self.j_frobenius_norm, nan=0.0))
            object.__setattr__(self, "j_frobenius_norm", max(0.0, val))

    @classmethod
    def create(
        cls,
        timestamp: float,
        state_3d: ManifoldState3D,
        jacobian_matrix: np.ndarray,
        state_4d: ManifoldState4D | None = None,
        divergence_threshold: float | None = None,
    ) -> ManifoldAuditRecord:
        """Factory method to calculate invariants and construct verified audit record."""
        thresh = divergence_threshold if divergence_threshold is not None else _DEFAULT_LIMITS.divergence_threshold
        J = np.asarray(jacobian_matrix, dtype=np.float64)
        div_val = float(np.trace(J)) if J.ndim == 2 else 0.0
        det_val = float(np.linalg.det(J)) if J.ndim == 2 and J.shape[0] == J.shape[1] else 0.0
        eigvals = np.linalg.eigvals(J) if J.ndim == 2 and J.shape[0] == J.shape[1] else np.zeros(3)
        max_eig = float(np.max(np.real(eigvals)))
        is_contr = div_val < 0.0
        is_4d = state_4d is not None

        if is_4d:
            verdict = "REGULARIZED_4D_GEODESIC"
        elif is_contr:
            verdict = "CONTRACTIVE_ATTRACTOR"
        elif div_val > thresh:
            verdict = "EXPANSIVE_CHAOS"
        else:
            verdict = "NEUTRAL_TRANSITION"

        frob_norm = float(np.linalg.norm(J, ord="fro")) if J.ndim == 2 else 0.0

        return cls(
            timestamp=float(timestamp),
            state_3d=state_3d,
            divergence=div_val,
            determinant=det_val,
            volume_rate=div_val,
            max_real_eigenvalue=max_eig,
            is_contractive=is_contr,
            is_4d_projected=is_4d,
            stability_verdict=verdict,
            state_4d=state_4d,
            j_frobenius_norm=frob_norm,
        )

    def to_telemetry_trace(self) -> dict[str, Any]:
        """Convert into ISO/IEC 22989 JSON-compliant telemetry trace dictionary."""
        trace: dict[str, Any] = {
            "timestamp": self.timestamp,
            "state_3d": self.state_3d.to_iso_audit_dict(),
            "divergence_tr_j": self.divergence,
            "determinant_det_j": self.determinant,
            "volume_rate_dv_dt": self.volume_rate,
            "max_real_eigenvalue": self.max_real_eigenvalue,
            "is_phase_volume_contractive": self.is_contractive,
            "is_4d_projected": self.is_4d_projected,
            "stability_verdict": self.stability_verdict,
            "j_frobenius_norm": self.j_frobenius_norm,
        }
        if self.state_4d is not None:
            trace["state_4d"] = self.state_4d.to_iso_audit_dict()
        return trace
