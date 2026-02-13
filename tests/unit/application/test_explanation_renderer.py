"""Tests para ExplanationRenderer.

Verifica que el renderer:
- No calcula lógica nueva (solo lee propiedades del dominio).
- Clasifica correctamente: certeza, desacuerdo, estabilidad, sobreajuste, conflicto.
- render_summary produce texto corto.
- render_technical_report produce reporte multi-sección.
- render_structured_json produce dict con bloque metacognitive.
"""

from __future__ import annotations

import json

import pytest

from iot_machine_learning.application.explainability.explanation_renderer import (
    ExplanationRenderer,
    _classify_certainty,
    _classify_cognitive_stability,
    _classify_disagreement,
    _classify_engine_conflict,
    _classify_overfit_risk,
)
from iot_machine_learning.domain.entities.explainability.contribution_breakdown import (
    ContributionBreakdown,
    EngineContribution,
)
from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
    Outcome,
)
from iot_machine_learning.domain.entities.explainability.reasoning_trace import (
    PhaseKind,
    ReasoningPhase,
    ReasoningTrace,
)
from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    FilterSnapshot,
    SignalSnapshot,
)


# ── Helpers ─────────────────────────────────────────────────────


def _engine(name: str, value: float, weight: float = 0.5,
            confidence: float = 0.8, trend: str = "up",
            inhibited: bool = False, reason: str = "none",
            stability: float = 0.1) -> EngineContribution:
    return EngineContribution(
        engine_name=name, predicted_value=value,
        confidence=confidence, trend=trend,
        base_weight=weight, final_weight=weight,
        inhibited=inhibited, inhibition_reason=reason,
        stability=stability,
    )


def _full_explanation(**overrides) -> Explanation:
    defaults = dict(
        series_id="temp_room_1",
        signal=SignalSnapshot(
            n_points=50, mean=20.0, std=2.0, noise_ratio=0.1,
            slope=0.5, curvature=-0.01, regime="stable", dt=1.0,
        ),
        filter=FilterSnapshot(
            filter_name="KalmanSignalFilter", n_points=50,
            noise_reduction_ratio=0.6, is_effective=True,
        ),
        contributions=ContributionBreakdown(
            contributions=[
                _engine("taylor", 25.0, weight=0.6, confidence=0.9),
                _engine("statistical", 23.0, weight=0.4, confidence=0.7),
            ],
            fusion_method="weighted_average",
            selected_engine="taylor",
            selection_reason="highest_weight",
        ),
        trace=ReasoningTrace(
            phases=[
                ReasoningPhase(kind=PhaseKind.PERCEIVE),
                ReasoningPhase(kind=PhaseKind.PREDICT),
                ReasoningPhase(kind=PhaseKind.FUSE),
            ],
            regime_at_inference="stable",
            n_engines_available=2,
            n_engines_active=2,
        ),
        outcome=Outcome(
            kind="prediction", predicted_value=24.2,
            confidence=0.85, trend="up",
        ),
    )
    defaults.update(overrides)
    return Explanation(**defaults)


# ── Classification functions ────────────────────────────────────


class TestClassifyCertainty:
    def test_high(self) -> None:
        assert _classify_certainty(0.90) == "high"

    def test_moderate(self) -> None:
        assert _classify_certainty(0.65) == "moderate"

    def test_low(self) -> None:
        assert _classify_certainty(0.40) == "low"

    def test_very_low(self) -> None:
        assert _classify_certainty(0.20) == "very_low"

    def test_boundary_high(self) -> None:
        assert _classify_certainty(0.85) == "high"

    def test_boundary_moderate(self) -> None:
        assert _classify_certainty(0.60) == "moderate"


class TestClassifyDisagreement:
    def test_single_engine(self) -> None:
        assert _classify_disagreement(0.0, 1) == "none"

    def test_consensus(self) -> None:
        assert _classify_disagreement(0.3, 3) == "consensus"

    def test_mild(self) -> None:
        assert _classify_disagreement(1.0, 3) == "mild"

    def test_significant(self) -> None:
        assert _classify_disagreement(3.0, 3) == "significant"

    def test_severe(self) -> None:
        assert _classify_disagreement(6.0, 3) == "severe"


class TestClassifyCognitiveStability:
    def test_stable(self) -> None:
        assert _classify_cognitive_stability(0, 3, False) == "stable"

    def test_adapting(self) -> None:
        assert _classify_cognitive_stability(0, 3, True) == "adapting"

    def test_stressed(self) -> None:
        assert _classify_cognitive_stability(2, 4, False) == "stressed"

    def test_degraded_all_inhibited(self) -> None:
        assert _classify_cognitive_stability(3, 3, False) == "degraded"

    def test_degraded_no_engines(self) -> None:
        assert _classify_cognitive_stability(0, 0, False) == "degraded"


class TestClassifyOverfitRisk:
    def test_single_engine(self) -> None:
        assert _classify_overfit_risk(1.0, 1) == "not_applicable"

    def test_high(self) -> None:
        assert _classify_overfit_risk(0.95, 3) == "high"

    def test_moderate(self) -> None:
        assert _classify_overfit_risk(0.75, 3) == "moderate"

    def test_low(self) -> None:
        assert _classify_overfit_risk(0.5, 3) == "low"


class TestClassifyEngineConflict:
    def test_single_engine(self) -> None:
        assert _classify_engine_conflict([
            _engine("a", 20.0, weight=1.0),
        ]) == "none"

    def test_aligned(self) -> None:
        assert _classify_engine_conflict([
            _engine("a", 20.0, trend="up"),
            _engine("b", 22.0, trend="up"),
        ]) == "aligned"

    def test_mild_divergence(self) -> None:
        assert _classify_engine_conflict([
            _engine("a", 20.0, trend="up"),
            _engine("b", 22.0, trend="stable"),
        ]) == "mild_divergence"

    def test_directional_conflict(self) -> None:
        assert _classify_engine_conflict([
            _engine("a", 20.0, trend="up"),
            _engine("b", 18.0, trend="down"),
        ]) == "directional_conflict"

    def test_inhibited_engines_excluded(self) -> None:
        """Engines with near-zero weight should not count."""
        assert _classify_engine_conflict([
            _engine("a", 20.0, weight=0.9, trend="up"),
            _engine("b", 18.0, weight=0.005, trend="down"),  # effectively off
        ]) == "none"


# ── render_summary ──────────────────────────────────────────────


class TestRenderSummary:
    def test_basic_summary(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation()
        summary = renderer.render_summary(exp)

        assert "24.2" in summary
        assert "high" in summary
        assert "taylor" in summary
        assert "stable" in summary

    def test_fallback_summary(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation(
            contributions=ContributionBreakdown(
                fallback_used=True,
                fallback_reason="no_valid_perceptions",
            ),
            outcome=Outcome(
                kind="prediction", predicted_value=20.0,
                confidence=0.2, trend="stable",
            ),
        )
        summary = renderer.render_summary(exp)
        assert "Fallback" in summary
        assert "no_valid_perceptions" in summary

    def test_minimal_explanation(self) -> None:
        renderer = ExplanationRenderer()
        exp = Explanation.minimal("s1")
        summary = renderer.render_summary(exp)
        # Should not crash, may be empty or minimal
        assert isinstance(summary, str)


# ── render_technical_report ─────────────────────────────────────


class TestRenderTechnicalReport:
    def test_has_all_sections(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation()
        report = renderer.render_technical_report(exp)

        assert "[Señal]" in report
        assert "[Filtrado]" in report
        assert "[Engines]" in report
        assert "[Metacognición]" in report
        assert "[Resultado]" in report
        assert "[Traza]" in report

    def test_metacognitive_labels_present(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation()
        report = renderer.render_technical_report(exp)

        assert "certeza:" in report
        assert "desacuerdo:" in report
        assert "estabilidad:" in report
        assert "sobreajuste:" in report
        assert "conflicto:" in report

    def test_inhibited_engine_marked(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation(
            contributions=ContributionBreakdown(
                contributions=[
                    _engine("taylor", 25.0, weight=0.9),
                    _engine("bad", 50.0, weight=0.1,
                            inhibited=True, reason="instability=0.9"),
                ],
                selected_engine="taylor",
            ),
        )
        report = renderer.render_technical_report(exp)
        assert "[INHIBIDO]" in report

    def test_no_filter_section_when_absent(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation(filter=FilterSnapshot.empty())
        report = renderer.render_technical_report(exp)
        assert "[Filtrado]" not in report

    def test_anomaly_in_report(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation(
            outcome=Outcome(
                kind="prediction+anomaly", predicted_value=25.0,
                confidence=0.7, trend="up",
                is_anomaly=True, anomaly_score=0.92,
            ),
        )
        report = renderer.render_technical_report(exp)
        assert "anomalía=sí" in report


# ── render_structured_json ──────────────────────────────────────


class TestRenderStructuredJson:
    def test_has_metacognitive_block(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation()
        result = renderer.render_structured_json(exp)

        assert "metacognitive" in result
        meta = result["metacognitive"]
        assert "certainty" in meta
        assert "disagreement" in meta
        assert "cognitive_stability" in meta
        assert "overfit_risk" in meta
        assert "engine_conflict" in meta

    def test_json_serializable(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation()
        result = renderer.render_structured_json(exp)

        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["metacognitive"]["certainty"] == "high"

    def test_preserves_base_explanation(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation()
        result = renderer.render_structured_json(exp)

        assert result["version"] == "1.0"
        assert result["series_id"] == "temp_room_1"
        assert "signal" in result
        assert "outcome" in result

    def test_high_disagreement_scenario(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation(
            contributions=ContributionBreakdown(
                contributions=[
                    _engine("taylor", 25.0, weight=0.5, trend="up"),
                    _engine("statistical", 15.0, weight=0.5, trend="down"),
                ],
                selected_engine="taylor",
            ),
            outcome=Outcome(
                kind="prediction", predicted_value=20.0,
                confidence=0.5, trend="up",
            ),
        )
        result = renderer.render_structured_json(exp)
        meta = result["metacognitive"]

        assert meta["certainty"] == "low"
        assert meta["disagreement"] == "severe"
        assert meta["engine_conflict"] == "directional_conflict"

    def test_degraded_scenario(self) -> None:
        renderer = ExplanationRenderer()
        exp = _full_explanation(
            contributions=ContributionBreakdown(
                contributions=[
                    _engine("a", 25.0, weight=0.9, inhibited=True,
                            reason="fit_error"),
                    _engine("b", 23.0, weight=0.05, inhibited=True,
                            reason="instability"),
                    _engine("c", 22.0, weight=0.05, inhibited=True,
                            reason="recent_error"),
                ],
                selected_engine="a",
            ),
            outcome=Outcome(
                kind="prediction", predicted_value=24.0,
                confidence=0.3, trend="stable",
            ),
        )
        result = renderer.render_structured_json(exp)
        meta = result["metacognitive"]

        assert meta["certainty"] == "very_low"
        assert meta["cognitive_stability"] == "degraded"
