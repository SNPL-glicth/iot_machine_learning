"""Unit tests for RootCauseDiagnosticService and decision decomposition.

Conforms to:
- ISO/IEC 22989:2022: Explainability and auditability.
- ISO/IEC 25010:2023: Analysability.
"""

from __future__ import annotations

from domain.services.explainability.root_cause_diagnostic_service import RootCauseDiagnosticService


def test_root_cause_diagnostic_nominal_regime() -> None:
    """Verify nominal case outputs NONE limiter and verified invariants."""
    service = RootCauseDiagnosticService()
    diag = service.decompose(
        i_mahalanobis=1.0, i_takens=1.0, i_risk_cvar=1.0,
        liouville_factor=0.98, sovereign_certainty=0.85, sovereign_polarity=1.0,
    )

    assert diag.dominant_limiter == "NONE"
    assert diag.prod_i_admissibility == 1.0
    assert "nominal" in diag.explanation_summary.lower()


def test_root_cause_diagnostic_mahalanobis_veto() -> None:
    """Verify Mahalanobis veto takes safety precedence."""
    service = RootCauseDiagnosticService()
    diag = service.decompose(
        i_mahalanobis=0.0, i_takens=1.0, i_risk_cvar=1.0,
        liouville_factor=1.0, sovereign_certainty=0.8, sovereign_polarity=1.0,
    )

    assert diag.dominant_limiter == "MAHALANOBIS"
    assert diag.prod_i_admissibility == 0.0
    assert "mahalanobis" in diag.explanation_summary.lower()


def test_root_cause_diagnostic_takens_veto() -> None:
    """Verify Takens FNN topological veto isolation."""
    service = RootCauseDiagnosticService()
    diag = service.decompose(
        i_mahalanobis=1.0, i_takens=0.0, i_risk_cvar=1.0,
        liouville_factor=1.0, sovereign_certainty=0.8, sovereign_polarity=1.0,
    )

    assert diag.dominant_limiter == "TAKENS_FNN"
    assert diag.prod_i_admissibility == 0.0
    assert "takens" in diag.explanation_summary.lower()


def test_root_cause_diagnostic_cvar_veto() -> None:
    """Verify CVaR risk admissibility veto isolation."""
    service = RootCauseDiagnosticService()
    diag = service.decompose(
        i_mahalanobis=1.0, i_takens=1.0, i_risk_cvar=0.0,
        liouville_factor=1.0, sovereign_certainty=0.8, sovereign_polarity=1.0,
    )

    assert diag.dominant_limiter == "CVAR_RISK"
    assert diag.prod_i_admissibility == 0.0
    assert "cvar" in diag.explanation_summary.lower()


def test_root_cause_diagnostic_liouville_attenuation() -> None:
    """Verify Liouville volume expansion damping isolation when admissibility is valid."""
    service = RootCauseDiagnosticService()
    diag = service.decompose(
        i_mahalanobis=1.0, i_takens=1.0, i_risk_cvar=1.0,
        liouville_factor=0.65, sovereign_certainty=0.9, sovereign_polarity=1.0,
    )

    assert diag.dominant_limiter == "LIOUVILLE_DISSIPATION"
    assert diag.liouville_suppression_pct > 30.0
    assert "liouville" in diag.explanation_summary.lower()


def test_root_cause_diagnostic_phase_opposition() -> None:
    """Verify Stokes phase opposition detection when polarity is negative."""
    service = RootCauseDiagnosticService()
    diag = service.decompose(
        i_mahalanobis=1.0, i_takens=1.0, i_risk_cvar=1.0,
        liouville_factor=0.95, sovereign_certainty=0.7, sovereign_polarity=-1.0,
    )

    assert diag.dominant_limiter == "PHASE_OPPOSITION"
    assert "polaridad" in diag.explanation_summary.lower()
