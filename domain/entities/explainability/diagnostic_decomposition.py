"""Domain entity for decision root-cause decomposition and AI explainability.

Conforms to:
- ISO/IEC 22989:2022: Information technology — Artificial intelligence (Explainability & Auditability).
- ISO/IEC 25010:2023: Software quality — Analysability and Diagnostic Integrity.
- Line count constraint: <= 180 lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DominantLimiter = Literal[
    "MAHALANOBIS",
    "TAKENS_FNN",
    "CVAR_RISK",
    "LIOUVILLE_DISSIPATION",
    "PHASE_OPPOSITION",
    "MOMENTUM_DEADBAND",
    "NONE",
]


@dataclass(frozen=True)
class DiagnosticDecomposition:
    """Immutable audit record detailing exact limiting factors of master equation."""

    timestamp_ns: int
    dominant_limiter: DominantLimiter
    prod_i_admissibility: float
    i_mahalanobis: float
    i_takens: float
    i_risk_cvar: float
    liouville_factor: float
    liouville_suppression_pct: float
    sovereign_certainty: float
    sovereign_polarity: float
    stokes_s0: float
    stokes_s1: float
    stokes_s3: float
    variable_destino: float
    explanation_summary: str
