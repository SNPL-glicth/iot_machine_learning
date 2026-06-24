"""Tests for fusion quality: FusePhase metrics, CoherenceCheckPhase context,
ConfidenceCalibrationPhase unification, and ReadinessGate engine_spread cap."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, Mock, PropertyMock

import pytest

from iot_machine_learning.infrastructure.ml.calibration import ConfidenceCalibrator
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.coherence_check_phase import (
    CROSS_REGIME_PENALTY,
    DRIFT_PENALTY,
    CoherenceCheckPhase,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.confidence_calibration_phase import (
    ConfidenceCalibrationPhase,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
    PipelineContext,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_perception(engine_name: str, predicted_value: float = 100.0, confidence: float = 0.85):
    p = MagicMock()
    p.engine_name = engine_name
    p.predicted_value = predicted_value
    p.confidence = confidence
    return p


def _make_ctx(**overrides) -> PipelineContext:
    class FakeTimer:
        total_ms = 10.0
        budget_ms = 500.0
        def to_dict(self):
            return {"total_ms": self.total_ms, "budget_ms": self.budget_ms}

    orch = MagicMock()
    orch._fusion = MagicMock()
    orch._fusion.fuse.return_value = (100.0, 0.85, "stable", {"a": 0.5, "b": 0.5}, "fusion", "consensus")
    orch._correlation_port = None
    orch._error_history = MagicMock()
    orch._error_history.get_error_dict_for_inhibition.return_value = {}

    profile = MagicMock()
    profile.std = 5.0
    profile.regime = "STABLE"
    profile.mean = 100.0
    profile.noise_ratio = 0.1
    profile.stability = 0.9
    profile.z_score = 0.0
    profile.series_state = "UNKNOWN"

    flags = MagicMock()
    flags.get.return_value = True

    ctx = PipelineContext(
        orchestrator=orch,
        values=[1.0, 2.0, 3.0],
        timestamps=[0.0, 1.0, 2.0],
        series_id="s1",
        flags=flags,
        timer=FakeTimer(),
        fused_value=100.0,
        fused_confidence=0.85,
        regime="STABLE",
        data_quality_score=1.0,
        cross_regime_incoherence=False,
        drift_detected=False,
        max_action="PREDICT",
        profile=profile,
    )
    ctx.neighbors = []
    ctx.neighbor_values = {}
    ctx.feature_context = None
    ctx.explanation = None
    ctx.inhibition_states = []
    ctx.neighbor_trends = None
    ctx.metrics_collector = None

    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


# ── ConfidenceCalibrator ───────────────────────────────────────────


class TestConfidenceCalibrator:
    def test_same_input_same_output(self):
        cal = ConfidenceCalibrator()
        r1 = cal.calibrate(score=0.85, regime="STABLE")
        r2 = cal.calibrate(score=0.85, regime="STABLE")
        assert r1.calibrated == r2.calibrated

    def test_floor_30(self):
        cal = ConfidenceCalibrator()
        result = cal.calibrate(score=-10.0, regime="STABLE")
        assert result.calibrated == 0.30

    def test_ceiling_95(self):
        cal = ConfidenceCalibrator()
        result = cal.calibrate(score=500.0, regime="STABLE")
        assert result.calibrated <= 0.95

    def test_data_quality_below_05_boosts_temperature(self):
        cal = ConfidenceCalibrator()
        good = cal.calibrate(score=0.85, regime="STABLE", data_quality=1.0)
        bad = cal.calibrate(score=0.85, regime="STABLE", data_quality=0.4)
        assert bad.calibrated <= good.calibrated
        assert any("data_quality" in r and "×1.3" in r for r in bad.reasons)

    def test_data_quality_below_03_boosts_temperature_more(self):
        cal = ConfidenceCalibrator()
        good = cal.calibrate(score=0.85, regime="STABLE", data_quality=1.0)
        bad = cal.calibrate(score=0.85, regime="STABLE", data_quality=0.2)
        assert bad.calibrated <= good.calibrated
        assert any("×1.6" in r for r in bad.reasons)

    def test_regime_temperature_override(self):
        cal = ConfidenceCalibrator()
        stable = cal.calibrate(score=0.85, regime="STABLE")
        volatile = cal.calibrate(score=0.85, regime="VOLATILE")
        assert volatile.calibrated <= stable.calibrated

    def test_invalid_score_returns_floor(self):
        cal = ConfidenceCalibrator()
        for bad in [float("nan"), float("inf"), float("-inf")]:
            result = cal.calibrate(score=bad)
            assert result.calibrated == 0.30


# ── CoherenceCheckPhase ────────────────────────────────────────────


class TestCoherenceCheckPhase:
    def test_no_penalties_when_context_clean(self):
        phase = CoherenceCheckPhase()
        ctx = _make_ctx(cross_regime_incoherence=False, drift_detected=False)
        result = phase.execute(ctx)
        assert result.fused_confidence == 0.85

    def test_cross_regime_penalty_applied(self):
        phase = CoherenceCheckPhase()
        ctx = _make_ctx(cross_regime_incoherence=True, drift_detected=False)
        result = phase.execute(ctx)
        expected = max(0.10, 0.85 - CROSS_REGIME_PENALTY)
        assert result.fused_confidence == pytest.approx(expected, abs=1e-4)
        assert result.coherence_result.penalties == ["cross_regime_incoherence"]

    def test_drift_penalty_applied(self):
        phase = CoherenceCheckPhase()
        ctx = _make_ctx(cross_regime_incoherence=False, drift_detected=True)
        result = phase.execute(ctx)
        expected = max(0.10, 0.85 - DRIFT_PENALTY)
        assert result.fused_confidence == pytest.approx(expected, abs=1e-4)
        assert "recent_drift" in result.coherence_result.penalties

    def test_both_penalties_stack(self):
        phase = CoherenceCheckPhase()
        ctx = _make_ctx(cross_regime_incoherence=True, drift_detected=True)
        result = phase.execute(ctx)
        expected = max(0.10, 0.85 - CROSS_REGIME_PENALTY - DRIFT_PENALTY)
        assert result.fused_confidence == pytest.approx(expected, abs=1e-4)
        assert len(result.coherence_result.penalties) == 2

    def test_drift_event_in_metadata_also_triggers(self):
        phase = CoherenceCheckPhase()
        ctx = _make_ctx(cross_regime_incoherence=False, drift_detected=False)
        ctx.metadata["drift_event"] = {"type": "abrupt", "magnitude": 1.5}
        result = phase.execute(ctx)
        expected = max(0.10, 0.85 - DRIFT_PENALTY)
        assert result.fused_confidence == pytest.approx(expected, abs=1e-4)

    def test_confidence_floor_observed(self):
        phase = CoherenceCheckPhase()
        ctx = _make_ctx(
            cross_regime_incoherence=True,
            drift_detected=True,
            fused_confidence=0.15,
        )
        result = phase.execute(ctx)
        assert result.fused_confidence >= 0.10


# ── FusePhase metrics (engine_spread → cap) ────────────────────────


class TestFusePhaseQualityMetrics:
    def test_engine_spread_and_rejected_in_metadata(self):
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            FusePhase,
        )

        phase = FusePhase()
        ctx = _make_ctx()
        os.environ["ML_HAMPEL_ENABLED"] = "false"

        ctx.perceptions = [
            _make_perception("taylor", 100.0),
            _make_perception("kalman", 110.0),
        ]

        result = phase.execute(ctx)
        assert "engine_spread" in result.metadata
        assert result.metadata["engine_spread"] > 0
        assert "spatial_correction_magnitude" in result.metadata

    def test_high_spread_caps_max_action(self):
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            FusePhase,
        )

        phase = FusePhase()
        ctx = _make_ctx()
        ctx.profile.std = 2.0
        ctx.max_action = "PREDICT"
        os.environ["ML_HAMPEL_ENABLED"] = "false"

        ctx.perceptions = [
            _make_perception("taylor", 100.0),
            _make_perception("kalman", 150.0),
        ]

        result = phase.execute(ctx)
        assert result.max_action == "INVESTIGATE"

    def test_low_spread_preserves_max_action(self):
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            FusePhase,
        )

        phase = FusePhase()
        ctx = _make_ctx()
        ctx.profile.std = 50.0
        ctx.max_action = "PREDICT"
        os.environ["ML_HAMPEL_ENABLED"] = "false"

        ctx.perceptions = [
            _make_perception("taylor", 100.0),
            _make_perception("kalman", 105.0),
        ]

        result = phase.execute(ctx)
        assert result.max_action == "PREDICT"
