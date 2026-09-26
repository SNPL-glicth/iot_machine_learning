"""Topological Audit Record Value Object for Takens Engine Explainability.

Conforms to:
- ISO/IEC 22989: Artificial Intelligence Transparency, Auditability, and Explainability.
- ISO/IEC 25010: Reliability and Numerical Fault Tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math


@dataclass(frozen=True, slots=True)
class TopologicalAuditRecord:
    """Immutable audit record logging topological reconstruction telemetry and veto decisions.

    Attributes:
        timestamp_ns: Event timestamp in nanoseconds for microsecond-precise correlation.
        d_effective: Participation ratio dimension D_eff of local delay covariance.
        fnn_ratio: False Nearest Neighbors ratio Omega_FNN measured on embedded trajectory.
        is_manifold_veto: Boolean flag indicating if topological critical veto was triggered.
        veto_reason: Structured diagnostic reason if vetoed, None if passed.
        manifold_coherence: Geodesic fidelity score Psi_takens(T) in [0.0, 1.0].
        embedded_norm: Euclidean norm ||y_t||_2 of the embedded state vector.
    """

    timestamp_ns: int
    d_effective: float
    fnn_ratio: float
    is_manifold_veto: bool
    veto_reason: str | None = None
    manifold_coherence: float = 1.0
    embedded_norm: float = 0.0

    def __post_init__(self) -> None:
        """Sanitize numerical telemetry values against NaNs and Infinities (ISO 25010)."""
        object.__setattr__(self, "timestamp_ns", max(0, int(self.timestamp_ns)))

        # Sanitize d_effective
        if not math.isfinite(self.d_effective):
            d_eff = 1.0
        else:
            d_eff = max(1.0, float(self.d_effective))
        object.__setattr__(self, "d_effective", d_eff)

        # Sanitize fnn_ratio
        if not math.isfinite(self.fnn_ratio):
            fnn = 0.0
        else:
            fnn = max(0.0, min(1.0, float(self.fnn_ratio)))
        object.__setattr__(self, "fnn_ratio", fnn)

        # Sanitize is_manifold_veto
        object.__setattr__(self, "is_manifold_veto", bool(self.is_manifold_veto))

        # Sanitize manifold_coherence in [0.0, 1.0]
        if not math.isfinite(self.manifold_coherence):
            coh = 0.0
        else:
            coh = max(0.0, min(1.0, float(self.manifold_coherence)))
        object.__setattr__(self, "manifold_coherence", coh)

        # Sanitize embedded_norm >= 0.0
        if not math.isfinite(self.embedded_norm):
            norm_val = 0.0
        else:
            norm_val = max(0.0, float(self.embedded_norm))
        object.__setattr__(self, "embedded_norm", norm_val)

        # Clean veto reason if no veto
        if not self.is_manifold_veto:
            object.__setattr__(self, "veto_reason", None)
        elif self.veto_reason is None:
            object.__setattr__(self, "veto_reason", "unspecified_topological_veto")

    def to_telemetry_trace(self) -> dict[str, Any]:
        """Serialize audit record to pure primitive dictionary for JSON/telemetry ingestion."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "d_effective": round(self.d_effective, 6),
            "fnn_ratio": round(self.fnn_ratio, 6),
            "is_manifold_veto": self.is_manifold_veto,
            "veto_reason": self.veto_reason,
            "manifold_coherence": round(self.manifold_coherence, 6),
            "embedded_norm": round(self.embedded_norm, 6),
        }
