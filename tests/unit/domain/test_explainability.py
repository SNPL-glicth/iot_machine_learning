"""Tests para la capa de explicabilidad cognitiva.

Cubre:
- SignalSnapshot / FilterSnapshot: construcción, serialización, from_dict.
- EngineContribution / ContributionBreakdown: métricas derivadas.
- ReasoningPhase / ReasoningTrace: fases, lookup, serialización.
- Explanation: composición, to_dict condicional, factory minimal.
- Outcome: serialización condicional.
"""

from __future__ import annotations

import json

import pytest

from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    FilterSnapshot,
    SignalSnapshot,
)
from iot_machine_learning.domain.entities.explainability.contribution_breakdown import (
    ContributionBreakdown,
    EngineContribution,
)
from iot_machine_learning.domain.entities.explainability.reasoning_trace import (
    PhaseKind,
    ReasoningPhase,
    ReasoningTrace,
    _safe_serialize,
)
from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
    Outcome,
)


# ── SignalSnapshot ──────────────────────────────────────────────


class TestSignalSnapshot:
    def test_empty(self) -> None:
        s = SignalSnapshot.empty()
        assert s.n_points == 0
        assert s.regime == "unknown"

    def test_to_dict_keys(self) -> None:
        s = SignalSnapshot(n_points=50, mean=20.0, std=2.0, noise_ratio=0.1,
                           slope=0.5, curvature=-0.01, regime="stable", dt=1.0)
        d = s.to_dict()
        assert d["n_points"] == 50
        assert d["regime"] == "stable"
        assert "extra" not in d  # no extra → no key

    def test_to_dict_with_extra(self) -> None:
        s = SignalSnapshot(extra={"trend_strength": 0.8})
        d = s.to_dict()
        assert d["extra"]["trend_strength"] == 0.8

    def test_from_dict_roundtrip(self) -> None:
        original = SignalSnapshot(n_points=30, mean=15.0, std=1.5,
                                  noise_ratio=0.1, slope=0.3, curvature=0.0,
                                  regime="trending", dt=2.0)
        d = original.to_dict()
        restored = SignalSnapshot.from_dict(d)
        assert restored.n_points == original.n_points
        assert restored.regime == original.regime
        assert restored.slope == pytest.approx(original.slope, abs=1e-5)

    def test_frozen(self) -> None:
        s = SignalSnapshot()
        with pytest.raises(AttributeError):
            s.n_points = 10  # type: ignore[misc]


# ── FilterSnapshot ──────────────────────────────────────────────


class TestFilterSnapshot:
    def test_empty(self) -> None:
        f = FilterSnapshot.empty()
        assert f.filter_name == "none"
        assert f.is_effective is False

    def test_to_dict(self) -> None:
        f = FilterSnapshot(filter_name="KalmanSignalFilter", n_points=100,
                           noise_reduction_ratio=0.6, is_effective=True)
        d = f.to_dict()
        assert d["filter_name"] == "KalmanSignalFilter"
        assert d["is_effective"] is True

    def test_from_dict_roundtrip(self) -> None:
        original = FilterSnapshot(filter_name="MedianSignalFilter",
                                  n_points=50, noise_reduction_ratio=0.3,
                                  lag_estimate=2, is_effective=True)
        d = original.to_dict()
        restored = FilterSnapshot.from_dict(d)
        assert restored.filter_name == original.filter_name
        assert restored.lag_estimate == original.lag_estimate


# ── EngineContribution ──────────────────────────────────────────


class TestEngineContribution:
    def test_weighted_contribution(self) -> None:
        c = EngineContribution(engine_name="taylor", predicted_value=25.0,
                               final_weight=0.6)
        assert c.weighted_contribution == pytest.approx(15.0)

    def test_to_dict_keys(self) -> None:
        c = EngineContribution(engine_name="ema", predicted_value=20.0,
                               confidence=0.8, inhibited=True,
                               inhibition_reason="high_instability")
        d = c.to_dict()
        assert d["engine_name"] == "ema"
        assert d["inhibited"] is True
        assert d["inhibition_reason"] == "high_instability"

    def test_frozen(self) -> None:
        c = EngineContribution(engine_name="x", predicted_value=0.0)
        with pytest.raises(AttributeError):
            c.engine_name = "y"  # type: ignore[misc]


# ── ContributionBreakdown ───────────────────────────────────────


class TestContributionBreakdown:
    def _make_breakdown(self) -> ContributionBreakdown:
        return ContributionBreakdown(
            contributions=[
                EngineContribution(engine_name="taylor", predicted_value=25.0,
                                   final_weight=0.6, confidence=0.9),
                EngineContribution(engine_name="statistical", predicted_value=23.0,
                                   final_weight=0.3, confidence=0.7),
                EngineContribution(engine_name="baseline", predicted_value=20.0,
                                   final_weight=0.1, confidence=0.5,
                                   inhibited=True,
                                   inhibition_reason="high_fit_error"),
            ],
            fusion_method="weighted_average",
            selected_engine="taylor",
            selection_reason="highest_weight",
        )

    def test_n_engines(self) -> None:
        b = self._make_breakdown()
        assert b.n_engines == 3

    def test_n_inhibited(self) -> None:
        b = self._make_breakdown()
        assert b.n_inhibited == 1

    def test_consensus_spread(self) -> None:
        b = self._make_breakdown()
        assert b.consensus_spread == pytest.approx(5.0)

    def test_consensus_spread_single_engine(self) -> None:
        b = ContributionBreakdown(contributions=[
            EngineContribution(engine_name="x", predicted_value=20.0),
        ])
        assert b.consensus_spread == 0.0

    def test_dominant_weight_ratio(self) -> None:
        b = self._make_breakdown()
        # 0.6 / (0.6 + 0.3 + 0.1) = 0.6
        assert b.dominant_weight_ratio == pytest.approx(0.6)

    def test_empty(self) -> None:
        b = ContributionBreakdown.empty()
        assert b.n_engines == 0
        assert b.dominant_weight_ratio == 0.0

    def test_to_dict_has_computed_fields(self) -> None:
        b = self._make_breakdown()
        d = b.to_dict()
        assert "n_engines" in d
        assert "n_inhibited" in d
        assert "consensus_spread" in d
        assert "dominant_weight_ratio" in d


# ── ReasoningPhase ──────────────────────────────────────────────


class TestReasoningPhase:
    def test_to_dict_minimal(self) -> None:
        p = ReasoningPhase(kind=PhaseKind.PERCEIVE,
                           summary={"n_points": 50, "regime": "stable"})
        d = p.to_dict()
        assert d["kind"] == "perceive"
        assert d["summary"]["n_points"] == 50
        assert "duration_ms" not in d  # optional, not set

    def test_to_dict_with_duration(self) -> None:
        p = ReasoningPhase(kind=PhaseKind.FUSE, duration_ms=1.234)
        d = p.to_dict()
        assert d["duration_ms"] == 1.234

    def test_safe_serialize_nested(self) -> None:
        """Objects with to_dict() are serialized recursively."""
        snap = SignalSnapshot(n_points=10, mean=5.0, std=1.0,
                              noise_ratio=0.2, slope=0.0, curvature=0.0)
        result = _safe_serialize({"signal": snap})
        assert isinstance(result["signal"], dict)
        assert result["signal"]["n_points"] == 10


# ── ReasoningTrace ──────────────────────────────────────────────


class TestReasoningTrace:
    def _make_trace(self) -> ReasoningTrace:
        return ReasoningTrace(
            phases=[
                ReasoningPhase(kind=PhaseKind.PERCEIVE,
                               summary={"regime": "stable"}),
                ReasoningPhase(kind=PhaseKind.FILTER,
                               summary={"filter": "Kalman"}),
                ReasoningPhase(kind=PhaseKind.PREDICT,
                               summary={"n_engines": 2}),
                ReasoningPhase(kind=PhaseKind.INHIBIT,
                               summary={"n_inhibited": 0}),
                ReasoningPhase(kind=PhaseKind.FUSE,
                               summary={"method": "weighted_average"}),
            ],
            regime_at_inference="stable",
            n_engines_available=3,
            n_engines_active=2,
        )

    def test_phase_kinds(self) -> None:
        t = self._make_trace()
        assert t.phase_kinds == ["perceive", "filter", "predict",
                                  "inhibit", "fuse"]

    def test_has_inhibition(self) -> None:
        t = self._make_trace()
        assert t.has_inhibition is True

    def test_has_adaptation_false(self) -> None:
        t = self._make_trace()
        assert t.has_adaptation is False

    def test_get_phase(self) -> None:
        t = self._make_trace()
        p = t.get_phase(PhaseKind.FILTER)
        assert p is not None
        assert p.summary["filter"] == "Kalman"

    def test_get_phase_missing(self) -> None:
        t = self._make_trace()
        assert t.get_phase(PhaseKind.ADAPT) is None

    def test_empty(self) -> None:
        t = ReasoningTrace.empty()
        assert len(t.phases) == 0
        assert t.phase_kinds == []

    def test_to_dict_serializable(self) -> None:
        t = self._make_trace()
        d = t.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ── Outcome ─────────────────────────────────────────────────────


class TestOutcome:
    def test_prediction_outcome(self) -> None:
        o = Outcome(kind="prediction", predicted_value=25.3,
                    confidence=0.85, trend="up")
        d = o.to_dict()
        assert d["predicted_value"] == pytest.approx(25.3, abs=1e-5)
        assert d["is_anomaly"] is False
        assert "anomaly_score" not in d

    def test_anomaly_outcome(self) -> None:
        o = Outcome(kind="anomaly", is_anomaly=True, anomaly_score=0.92)
        d = o.to_dict()
        assert d["is_anomaly"] is True
        assert d["anomaly_score"] == pytest.approx(0.92, abs=1e-5)

    def test_combined_outcome(self) -> None:
        o = Outcome(kind="prediction+anomaly", predicted_value=25.0,
                    confidence=0.7, is_anomaly=True, anomaly_score=0.8)
        d = o.to_dict()
        assert "predicted_value" in d
        assert "anomaly_score" in d


# ── Explanation ─────────────────────────────────────────────────


class TestExplanation:
    def test_minimal(self) -> None:
        e = Explanation.minimal("temp_room_1")
        assert e.series_id == "temp_room_1"
        assert e.has_filter_data is False
        assert e.has_trace is False
        assert e.has_contributions is False

    def test_minimal_to_dict(self) -> None:
        e = Explanation.minimal("s1")
        d = e.to_dict()
        assert d["version"] == "1.0"
        assert d["series_id"] == "s1"
        assert "signal" in d
        assert "outcome" in d
        # Conditional sections omitted when empty
        assert "filter" not in d
        assert "contributions" not in d
        assert "trace" not in d

    def test_full_explanation_to_dict(self) -> None:
        e = Explanation(
            series_id="sensor_42",
            signal=SignalSnapshot(n_points=50, mean=20.0, std=2.0,
                                  noise_ratio=0.1, slope=0.5,
                                  curvature=-0.01, regime="stable"),
            filter=FilterSnapshot(filter_name="KalmanSignalFilter",
                                  n_points=50, noise_reduction_ratio=0.6,
                                  is_effective=True),
            contributions=ContributionBreakdown(
                contributions=[
                    EngineContribution(engine_name="taylor",
                                       predicted_value=25.0,
                                       final_weight=0.7),
                ],
                selected_engine="taylor",
            ),
            trace=ReasoningTrace(
                phases=[
                    ReasoningPhase(kind=PhaseKind.PERCEIVE,
                                   summary={"regime": "stable"}),
                ],
                regime_at_inference="stable",
            ),
            outcome=Outcome(kind="prediction", predicted_value=25.0,
                            confidence=0.85, trend="up"),
            audit_trace_id="abc-123",
        )

        d = e.to_dict()

        # All sections present
        assert "signal" in d
        assert "filter" in d
        assert "contributions" in d
        assert "trace" in d
        assert "audit_trace_id" in d
        assert d["audit_trace_id"] == "abc-123"

        # JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_has_filter_data(self) -> None:
        e = Explanation(
            series_id="s1",
            filter=FilterSnapshot(filter_name="EMASignalFilter"),
        )
        assert e.has_filter_data is True

    def test_has_trace(self) -> None:
        e = Explanation(
            series_id="s1",
            trace=ReasoningTrace(phases=[
                ReasoningPhase(kind=PhaseKind.FUSE),
            ]),
        )
        assert e.has_trace is True
        assert e.n_phases == 1

    def test_frozen(self) -> None:
        e = Explanation.minimal("s1")
        with pytest.raises(AttributeError):
            e.series_id = "s2"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        """Full explanation survives JSON serialization."""
        e = Explanation(
            series_id="test",
            signal=SignalSnapshot(n_points=10, mean=5.0, std=1.0,
                                  noise_ratio=0.2, slope=0.1,
                                  curvature=0.0, regime="idle"),
            outcome=Outcome(kind="prediction", predicted_value=5.5,
                            confidence=0.9),
        )
        d = e.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["series_id"] == "test"
        assert parsed["signal"]["regime"] == "idle"
        assert parsed["outcome"]["predicted_value"] == pytest.approx(5.5, abs=1e-4)
