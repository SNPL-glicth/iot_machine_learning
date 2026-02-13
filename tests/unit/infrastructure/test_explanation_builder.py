"""Tests para ExplanationBuilder y su integración con el orquestador.

Cubre:
- Builder standalone: fases dinámicas, no pipeline fijo.
- Solo PERCEIVE cuando no hay engines.
- PREDICT solo si engines respondieron.
- ADAPT solo si plasticidad participó.
- INHIBIT solo si algún engine fue suprimido.
- Fallback path.
- Integración end-to-end con MetaCognitiveOrchestrator.
- Explanation en metadata de PredictionResult.
- JSON serializable.
"""

from __future__ import annotations

import json

import pytest

from iot_machine_learning.domain.entities.explainability.explanation import Explanation
from iot_machine_learning.domain.entities.explainability.reasoning_trace import PhaseKind
from iot_machine_learning.infrastructure.ml.cognitive.explanation_builder import (
    ExplanationBuilder,
)
from iot_machine_learning.domain.entities.series.structural_analysis import (
    RegimeType,
    StructuralAnalysis,
)
from iot_machine_learning.infrastructure.ml.cognitive.types import (
    EnginePerception,
    InhibitionState,
)


def _make_profile(**overrides) -> StructuralAnalysis:
    defaults = dict(
        n_points=50, mean=20.0, std=2.0, noise_ratio=0.1,
        slope=0.5, curvature=-0.01, regime=RegimeType.STABLE, dt=1.0,
    )
    defaults.update(overrides)
    return StructuralAnalysis(**defaults)


def _make_perception(name: str, value: float, **kw) -> EnginePerception:
    return EnginePerception(
        engine_name=name, predicted_value=value,
        confidence=kw.get("confidence", 0.8),
        trend=kw.get("trend", "up"),
        stability=kw.get("stability", 0.1),
        local_fit_error=kw.get("local_fit_error", 0.5),
    )


def _make_inh(name: str, bw: float, iw: float, reason: str = "none") -> InhibitionState:
    return InhibitionState(
        engine_name=name, base_weight=bw, inhibited_weight=iw,
        inhibition_reason=reason,
        suppression_factor=max(0.0, 1.0 - iw / bw) if bw > 0 else 0.0,
    )


# ── Dynamic phases ──────────────────────────────────────────────


class TestDynamicPhases:
    """Phases are added dynamically, not as a fixed pipeline."""

    def test_perceive_only(self) -> None:
        """Only PERCEIVE when no engines respond."""
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())
        exp = builder.build()

        assert exp.has_trace
        assert exp.trace.phase_kinds == ["perceive"]

    def test_perceive_predict_fuse(self) -> None:
        """No inhibition, no adaptation → only PERCEIVE + PREDICT + FUSE."""
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())

        perceptions = [_make_perception("taylor", 25.0)]
        builder.set_perceptions(perceptions, n_engines_total=1)

        # No adaptation
        builder.set_adaptation(adapted=False, regime="stable")

        # No inhibition (suppression_factor = 0)
        inh = [_make_inh("taylor", 1.0, 1.0)]
        builder.set_inhibition(inh, {"taylor": 1.0})

        builder.set_fusion(25.0, 0.8, "up", {"taylor": 1.0},
                           "taylor", "highest_weight", "single_engine")

        exp = builder.build()
        assert exp.trace.phase_kinds == ["perceive", "predict", "fuse"]
        assert not exp.trace.has_inhibition
        assert not exp.trace.has_adaptation

    def test_full_pipeline_with_inhibition_and_adaptation(self) -> None:
        """All phases present when inhibition and adaptation fire."""
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())

        perceptions = [
            _make_perception("taylor", 25.0),
            _make_perception("statistical", 23.0, stability=0.8),
        ]
        builder.set_perceptions(perceptions, n_engines_total=2)
        builder.set_adaptation(adapted=True, regime="stable")

        inh = [
            _make_inh("taylor", 0.5, 0.5),
            _make_inh("statistical", 0.5, 0.1, "instability=0.800"),
        ]
        builder.set_inhibition(inh, {"taylor": 0.5, "statistical": 0.5})

        builder.set_fusion(24.5, 0.75, "up",
                           {"taylor": 0.83, "statistical": 0.17},
                           "taylor", "highest_weight", "weighted_average")

        exp = builder.build()
        assert exp.trace.phase_kinds == [
            "perceive", "predict", "adapt", "inhibit", "fuse"
        ]
        assert exp.trace.has_inhibition
        assert exp.trace.has_adaptation

    def test_filter_phase_added_when_set(self) -> None:
        """FILTER phase only present when set_filter is called."""
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())
        builder.set_filter("KalmanSignalFilter", {
            "n_points": 50, "noise_reduction_ratio": 0.6,
            "signal_distortion": 0.02,
        })

        exp = builder.build()
        assert "filter" in exp.trace.phase_kinds
        assert exp.has_filter_data
        assert exp.filter.filter_name == "KalmanSignalFilter"

    def test_no_filter_phase_when_not_set(self) -> None:
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())
        exp = builder.build()
        assert "filter" not in exp.trace.phase_kinds
        assert not exp.has_filter_data


# ── ContributionBreakdown ───────────────────────────────────────


class TestContributions:
    """Contributions are populated correctly from perceptions + inhibition."""

    def test_contributions_populated(self) -> None:
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())

        perceptions = [
            _make_perception("taylor", 25.0, confidence=0.9),
            _make_perception("statistical", 23.0, confidence=0.7),
        ]
        builder.set_perceptions(perceptions, n_engines_total=2)

        inh = [
            _make_inh("taylor", 0.5, 0.5),
            _make_inh("statistical", 0.5, 0.5),
        ]
        builder.set_inhibition(inh, {"taylor": 0.5, "statistical": 0.5})

        builder.set_fusion(24.0, 0.8, "up",
                           {"taylor": 0.5, "statistical": 0.5},
                           "taylor", "highest_weight")

        exp = builder.build()
        assert exp.has_contributions
        assert exp.contributions.n_engines == 2
        assert exp.contributions.selected_engine == "taylor"

        # Check individual contributions
        taylor_c = next(
            c for c in exp.contributions.contributions
            if c.engine_name == "taylor"
        )
        assert taylor_c.predicted_value == 25.0
        assert taylor_c.final_weight == pytest.approx(0.5)

    def test_inhibited_engine_marked(self) -> None:
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())

        perceptions = [
            _make_perception("taylor", 25.0),
            _make_perception("bad_engine", 50.0, stability=0.9),
        ]
        builder.set_perceptions(perceptions, n_engines_total=2)

        inh = [
            _make_inh("taylor", 0.5, 0.5),
            _make_inh("bad_engine", 0.5, 0.05, "instability=0.900"),
        ]
        builder.set_inhibition(inh, {"taylor": 0.5, "bad_engine": 0.5})

        builder.set_fusion(25.0, 0.8, "up",
                           {"taylor": 0.91, "bad_engine": 0.09},
                           "taylor", "highest_weight")

        exp = builder.build()
        bad = next(
            c for c in exp.contributions.contributions
            if c.engine_name == "bad_engine"
        )
        assert bad.inhibited is True
        assert bad.inhibition_reason == "instability=0.900"
        assert exp.contributions.n_inhibited == 1


# ── Fallback ────────────────────────────────────────────────────


class TestFallback:
    def test_fallback_explanation(self) -> None:
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())
        builder.set_fallback(20.0, reason="no_valid_perceptions")

        exp = builder.build()
        assert exp.contributions.fallback_used is True
        assert exp.contributions.fallback_reason == "no_valid_perceptions"
        assert exp.outcome.predicted_value == pytest.approx(20.0)
        assert exp.outcome.confidence == pytest.approx(0.2)
        assert exp.contributions.n_engines == 0


# ── Outcome ─────────────────────────────────────────────────────


class TestOutcome:
    def test_outcome_from_fusion(self) -> None:
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())
        perceptions = [_make_perception("taylor", 25.0)]
        builder.set_perceptions(perceptions)
        inh = [_make_inh("taylor", 1.0, 1.0)]
        builder.set_inhibition(inh, {"taylor": 1.0})
        builder.set_fusion(25.0, 0.85, "up", {"taylor": 1.0},
                           "taylor", "single", "single_engine")

        exp = builder.build()
        assert exp.outcome.kind == "prediction"
        assert exp.outcome.predicted_value == pytest.approx(25.0)
        assert exp.outcome.confidence == pytest.approx(0.85)
        assert exp.outcome.trend == "up"


# ── Serialization ───────────────────────────────────────────────


class TestSerialization:
    def test_full_explanation_json_serializable(self) -> None:
        builder = ExplanationBuilder("sensor_42")
        builder.set_signal(_make_profile())
        builder.set_filter("KalmanSignalFilter", {
            "n_points": 50, "noise_reduction_ratio": 0.6,
        })
        perceptions = [
            _make_perception("taylor", 25.0),
            _make_perception("statistical", 23.0),
        ]
        builder.set_perceptions(perceptions, n_engines_total=3)
        builder.set_adaptation(adapted=True, regime="stable")
        inh = [
            _make_inh("taylor", 0.5, 0.5),
            _make_inh("statistical", 0.5, 0.5),
        ]
        builder.set_inhibition(inh, {"taylor": 0.5, "statistical": 0.5})
        builder.set_fusion(24.0, 0.8, "up",
                           {"taylor": 0.5, "statistical": 0.5},
                           "taylor", "highest_weight")
        builder.set_audit_trace_id("audit-123")

        exp = builder.build()
        d = exp.to_dict()

        # Must be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        assert parsed["version"] == "1.0"
        assert parsed["series_id"] == "sensor_42"
        assert parsed["audit_trace_id"] == "audit-123"
        assert "signal" in parsed
        assert "filter" in parsed
        assert "contributions" in parsed
        assert "trace" in parsed
        assert "outcome" in parsed

    def test_minimal_explanation_json(self) -> None:
        builder = ExplanationBuilder("s1")
        builder.set_signal(_make_profile())
        exp = builder.build()
        d = exp.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ── Orchestrator integration ────────────────────────────────────


class TestOrchestratorIntegration:
    """Verify the orchestrator produces Explanation in metadata."""

    def _make_engine(self, name: str, value: float, confidence: float = 0.8):
        """Create a minimal mock engine."""
        from iot_machine_learning.infrastructure.ml.interfaces import (
            PredictionEngine,
            PredictionResult,
        )

        class _MockEngine(PredictionEngine):
            @property
            def name(self) -> str:
                return name

            def can_handle(self, n_points: int) -> bool:
                return n_points >= 3

            def predict(self, values, timestamps=None):
                return PredictionResult(
                    predicted_value=value,
                    confidence=confidence,
                    trend="up",
                    metadata={"diagnostic": {
                        "stability_indicator": 0.1,
                        "local_fit_error": 0.3,
                    }},
                )

        return _MockEngine()

    def test_predict_produces_explanation(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestrator import (
            MetaCognitiveOrchestrator,
        )

        orch = MetaCognitiveOrchestrator(
            engines=[self._make_engine("eng_a", 25.0)],
            enable_plasticity=False,
        )

        result = orch.predict([20.0, 21.0, 22.0, 23.0, 24.0])

        # Explanation in metadata
        assert "explanation" in result.metadata
        exp_dict = result.metadata["explanation"]
        assert exp_dict["version"] == "1.0"
        assert "signal" in exp_dict
        assert "outcome" in exp_dict

        # Also accessible via property
        exp = orch.last_explanation
        assert exp is not None
        assert isinstance(exp, Explanation)
        assert exp.outcome.predicted_value == pytest.approx(25.0)

    def test_predict_explanation_has_dynamic_phases(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestrator import (
            MetaCognitiveOrchestrator,
        )

        orch = MetaCognitiveOrchestrator(
            engines=[
                self._make_engine("eng_a", 25.0),
                self._make_engine("eng_b", 23.0, confidence=0.6),
            ],
            enable_plasticity=False,
        )

        orch.predict([20.0, 21.0, 22.0, 23.0, 24.0])
        exp = orch.last_explanation

        # Should have PERCEIVE, PREDICT, FUSE (no ADAPT since plasticity off)
        assert "perceive" in exp.trace.phase_kinds
        assert "predict" in exp.trace.phase_kinds
        assert "fuse" in exp.trace.phase_kinds
        assert not exp.trace.has_adaptation

    def test_fallback_produces_explanation(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestrator import (
            MetaCognitiveOrchestrator,
        )

        # Engine that can't handle small windows
        eng = self._make_engine("eng_a", 25.0)

        orch = MetaCognitiveOrchestrator(engines=[eng])

        # Only 2 points — engine can't handle (needs >= 3)
        result = orch.predict([20.0, 21.0])

        assert "explanation" in result.metadata
        exp = orch.last_explanation
        assert exp.contributions.fallback_used is True
        assert exp.trace.phase_kinds == ["perceive"]

    def test_explanation_json_roundtrip_from_orchestrator(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestrator import (
            MetaCognitiveOrchestrator,
        )

        orch = MetaCognitiveOrchestrator(
            engines=[self._make_engine("eng_a", 25.0)],
            enable_plasticity=False,
        )

        result = orch.predict([20.0, 21.0, 22.0, 23.0, 24.0])
        exp_dict = result.metadata["explanation"]

        json_str = json.dumps(exp_dict)
        parsed = json.loads(json_str)
        assert parsed["series_id"] == "unknown"
        assert parsed["outcome"]["predicted_value"] == pytest.approx(25.0, abs=1e-4)

    def test_series_id_passed_through(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestrator import (
            MetaCognitiveOrchestrator,
        )

        orch = MetaCognitiveOrchestrator(
            engines=[self._make_engine("eng_a", 25.0)],
            enable_plasticity=False,
        )

        result = orch.predict(
            [20.0, 21.0, 22.0, 23.0, 24.0],
            series_id="temp_room_1",
        )

        exp = orch.last_explanation
        assert exp.series_id == "temp_room_1"
        assert result.metadata["explanation"]["series_id"] == "temp_room_1"
