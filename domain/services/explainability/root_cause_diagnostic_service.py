"""Root-Cause Diagnostic and AI Explainability Domain Service.

Conforms to:
- ISO/IEC 22989:2022: Information technology — Artificial intelligence (Explainability).
- ISO/IEC 25010:2023: Systems and software Quality — Analysability.
- Strict constraint: Line count <= 180 lines, pure domain logic.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from domain.entities.explainability.diagnostic_decomposition import (
    DiagnosticDecomposition,
    DominantLimiter,
)


class RootCauseDiagnosticService:
    """Decomposes sovereign master decisions into exact limiting physical factors."""

    def decompose(
        self,
        i_mahalanobis: float,
        i_takens: float,
        i_risk_cvar: float,
        liouville_factor: float = 1.0,
        sovereign_certainty: float = 1.0,
        sovereign_polarity: float = 1.0,
        stokes_s0: float = 1.0,
        stokes_s1: float = 0.0,
        stokes_s3: float = 1.0,
        variable_destino: float = 1.0,
        momentum_veto: float = 1.0,
        timestamp_ns: int | None = None,
    ) -> DiagnosticDecomposition:
        """Analyze multi-layer invariants and return deterministic diagnostic record."""
        t_ns = time.time_ns() if timestamp_ns is None else timestamp_ns

        im = float(i_mahalanobis)
        it = float(i_takens)
        ir = float(i_risk_cvar)
        prod_i = float(im * it * ir)

        lf = float(liouville_factor)
        supp_pct = float(max(0.0, (1.0 - lf) * 100.0))
        sc = float(sovereign_certainty)
        sp = float(sovereign_polarity)

        # 1. Evaluate dominant limiter strictly in hierarchy of safety
        limiter: DominantLimiter = "NONE"
        explanation = "Régimen nominal continuo: todos los invariantes de estabilidad verificados."

        if im <= 0.0:
            limiter = "MAHALANOBIS"
            explanation = "Veto elíptico de Mahalanobis: perturbación anómala detectada en módulo de ingesta."
        elif it <= 0.0:
            limiter = "TAKENS_FNN"
            explanation = "Veto topológico de Takens: falsos vecinos más cercanos superaron umbral crítico de pliegue."
        elif ir <= 0.0:
            limiter = "CVAR_RISK"
            explanation = "Veto estocástico de riesgo: déficit en la admisibilidad de cola CVaR."
        elif lf < 0.80:
            limiter = "LIOUVILLE_DISSIPATION"
            explanation = f"Amortiguamiento de Liouville activo: expansión de volumen de fases atenuó certeza {supp_pct:.1f}%."
        elif sp < 0.0:
            limiter = "PHASE_OPPOSITION"
            explanation = "Inversión de polaridad soberana: interferencia destructiva de Stokes (dominio del modo conjugado MRT)."
        elif float(momentum_veto) <= 0.0:
            limiter = "MOMENTUM_DEADBAND"
            explanation = "Veto inercial: velocidad de fase dentro de la banda muerta del ruido del sistema."

        return DiagnosticDecomposition(
            timestamp_ns=t_ns,
            dominant_limiter=limiter,
            prod_i_admissibility=prod_i,
            i_mahalanobis=im,
            i_takens=it,
            i_risk_cvar=ir,
            liouville_factor=lf,
            liouville_suppression_pct=supp_pct,
            sovereign_certainty=sc,
            sovereign_polarity=sp,
            stokes_s0=float(stokes_s0),
            stokes_s1=float(stokes_s1),
            stokes_s3=float(stokes_s3),
            variable_destino=float(variable_destino),
            explanation_summary=explanation,
        )

    def decompose_from_components(
        self,
        components: Any,
        timestamp_ns: int | None = None,
    ) -> DiagnosticDecomposition:
        """Extract diagnostic factors directly from MasterEquationComponents object."""
        # Safe introspection of domain DTO
        geo = getattr(components, "geometric_manifold_shadow", {}) or {}
        stokes_s0 = float(geo.get("stokes_s0", 1.0))
        stokes_s1 = float(geo.get("stokes_s1", 0.0))
        stokes_s3 = float(geo.get("stokes_s3", 1.0))

        prod_i = float(getattr(components, "i_admissibility", 1.0))
        # If composite admissibility is 1.0, individual factors are 1.0
        im = 1.0 if prod_i > 0.0 else 0.0
        it = 1.0 if prod_i > 0.0 else 0.0
        ir = 1.0 if getattr(components, "i_cvar", 1.0) > 0.0 else 0.0

        lf = float(geo.get("liouville_factor", 1.0))
        sc = float(getattr(components, "sovereign_certainty", 1.0))
        sp = float(getattr(components, "sovereign_polarity", 1.0))
        vd = float(getattr(components, "variable_destino", 1.0))
        mv = float(getattr(components, "momentum_veto", 1.0))

        return self.decompose(
            i_mahalanobis=im,
            i_takens=it,
            i_risk_cvar=ir,
            liouville_factor=lf,
            sovereign_certainty=sc,
            sovereign_polarity=sp,
            stokes_s0=stokes_s0,
            stokes_s1=stokes_s1,
            stokes_s3=stokes_s3,
            variable_destino=vd,
            momentum_veto=mv,
            timestamp_ns=timestamp_ns,
        )
