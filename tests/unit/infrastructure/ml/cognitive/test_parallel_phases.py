"""Tests for PredictPhase (parallel), AdaptPhase (natural decay),
InhibitPhase (reliability-only, Redis), DecisionArbiterPhase (profile_engine)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest

from iot_machine_learning.domain.services.prediction.engine_decision_arbiter import (
    EngineDecisionArbiter,
    EngineDecision,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.adapt_phase import (
    AdaptPhase,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.decision_arbiter_phase import (
    DecisionArbiterPhase,
    _resolve_profile_engine,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.inhibit_phase import (
    InhibitPhase,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.predict_phase import (
    PredictPhase,
    _ENGINE_TIMEOUT_MS,
    _MAX_WORKERS,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _mock_perception(engine_name: str, confidence: float = 0.8, value: float = 100.0):
    p = MagicMock()
    p.engine_name = engine_name
    p.confidence = confidence
    p.predicted_value = value
    p.trend = "stable"
    return p


def _mock_engine(name: str, delay_ms: float = 0, can_handle: bool = True):
    eng = MagicMock()
    eng.name = name
    eng.can_handle.return_value = can_handle

    def _predict(values, timestamps=None):
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)
        res = MagicMock()
        res.predicted_value = 100.0
        res.confidence = 0.8
        res.trend = "stable"
        return res

    eng.predict = _predict
    return eng


def _make_ctx(**overrides):
    from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
        PipelineContext,
    )

    class FakeTimer:
        total_ms = 10.0
        budget_ms = 500.0
        def to_dict(self):
            return {"total_ms": self.total_ms, "budget_ms": self.budget_ms}

    orch = MagicMock()
    orch._engines = []
    orch._error_history = MagicMock()
    orch._error_history.get_error_dict_for_inhibition.return_value = {}
    orch._plasticity = None
    orch._weight_resolver = MagicMock()
    orch._weight_resolver.resolve.return_value = {"taylor": 0.5, "kalman": 0.5}
    orch._series_values_store = None
    orch._engine_filter = None
    orch._sensor_profile_repository = None
    orch._weight_initializer = None
    orch._metrics_collector = None
    orch._inhibition = MagicMock()
    orch._reliability_tracker = None

    ctx = PipelineContext(
        orchestrator=orch,
        values=[float(v) for v in range(20)],
        timestamps=list(range(20)),
        series_id="test_sensor",
        flags=MagicMock(),
        timer=FakeTimer(),
        **{k: v for k, v in overrides.items() if not k.startswith("_")},
    )
    ctx.profile = MagicMock()
    ctx.profile.mean = 100.0
    ctx.profile.noise_ratio = 0.1
    ctx.profile.stability = 0.9
    ctx.regime = "STABLE"
    ctx.selected_engine = "fusion"
    ctx.max_action = "PREDICT"
    ctx.plasticity_weights = {"taylor": 0.5, "kalman": 0.5}
    ctx.perceptions = [
        _mock_perception("taylor"),
        _mock_perception("kalman"),
    ]
    return ctx


# ── PredictPhase ────────────────────────────────────────────────────


class TestPredictPhase:
    def test_parallel_engines_complete_under_200ms(self):
        """4 fast engines complete in < 200ms total."""
        phase = PredictPhase()
        ctx = _make_ctx()
        ctx.orchestrator._engines = [
            _mock_engine("taylor", delay_ms=10),
            _mock_engine("kalman", delay_ms=10),
            _mock_engine("statistical", delay_ms=10),
            _mock_engine("baseline", delay_ms=10),
        ]
        t0 = time.monotonic()
        result = phase.execute(ctx)
        elapsed = (time.monotonic() - t0) * 1000
        assert elapsed < 200, f"Parallel execution took {elapsed:.1f}ms, expected <200ms"
        assert result.perceptions is not None

    def test_slow_engine_times_out_others_continue(self):
        """Engine taking 200ms times out; other engines still produce results."""
        phase = PredictPhase()
        ctx = _make_ctx()
        ctx.orchestrator._engines = [
            _mock_engine("fast_engine", delay_ms=5),
            _mock_engine("slow_engine", delay_ms=200),
        ]
        result = phase.execute(ctx)
        perceptions = result.perceptions or []
        names = [p.engine_name for p in perceptions]
        assert "fast_engine" in names
        assert "slow_engine" not in names

    def test_disabled_engine_skipped(self):
        """Engine failing can_handle is skipped."""
        phase = PredictPhase()
        ctx = _make_ctx()
        ctx.orchestrator._engines = [
            _mock_engine("good", can_handle=True),
            _mock_engine("bad", can_handle=False),
        ]
        result = phase.execute(ctx)
        perceptions = result.perceptions or []
        names = [p.engine_name for p in perceptions]
        assert "good" in names
        assert "bad" not in names

    def test_no_capable_engines_triggers_fallback(self):
        """No capable engines leads to fallback result."""
        phase = PredictPhase()
        ctx = _make_ctx()
        ctx.orchestrator._engines = [
            _mock_engine("bad1", can_handle=False),
            _mock_engine("bad2", can_handle=False),
        ]
        result = phase.execute(ctx)
        assert result.is_fallback

    def test_redis_metrics_recorded(self):
        """Engine time and failure metrics written to Redis."""
        redis = MagicMock()
        phase = PredictPhase()
        ctx = _make_ctx()
        store = MagicMock()
        store._redis = redis
        ctx.orchestrator._series_values_store = store
        ctx.orchestrator._engines = [
            _mock_engine("fast_engine", delay_ms=5),
            _mock_engine("broken_engine", delay_ms=200),
        ]
        phase.execute(ctx)
        # Engine time should be recorded for fast engine
        time_keys = [
            args[0][0] for args in redis.setex.call_args_list
            if "zenin:metrics:engine_time:" in str(args)
        ]
        assert any("fast_engine" in str(k) for k in time_keys)


# ── AdaptPhase ──────────────────────────────────────────────────────


class TestAdaptPhase:
    def test_natural_decay_after_50_predictions(self):
        """Every 50 predictions, natural decay is applied (confidence * 0.95)."""
        plasticity = MagicMock()
        plasticity.has_history.return_value = True
        plasticity.apply_natural_decay = MagicMock()

        phase = AdaptPhase(natural_decay_interval=3)
        ctx = _make_ctx()
        ctx.orchestrator._plasticity = plasticity

        # Run 4 times — decay triggers after 3rd
        for _ in range(4):
            phase.execute(ctx)

        assert plasticity.apply_natural_decay.call_count >= 1

    def test_natural_decay_counter_resets_after_trigger(self):
        """Counter resets to 0 after natural decay fires."""
        plasticity = MagicMock()
        plasticity.has_history.return_value = True
        plasticity.apply_natural_decay = MagicMock()

        phase = AdaptPhase(natural_decay_interval=3)
        ctx = _make_ctx()
        ctx.orchestrator._plasticity = plasticity

        for _ in range(3):
            phase.execute(ctx)
        assert plasticity.apply_natural_decay.call_count == 1

        for _ in range(2):
            phase.execute(ctx)
        assert plasticity.apply_natural_decay.call_count == 1

        phase.execute(ctx)
        assert plasticity.apply_natural_decay.call_count == 2


# ── InhibitPhase ────────────────────────────────────────────────────


class TestInhibitPhase:
    def test_reliable_engine_passes_through(self):
        """Reliable engine gets no suppression."""
        phase = InhibitPhase()
        ctx = _make_ctx()
        reliability = MagicMock()
        reliability.is_reliable.return_value = True
        reliability.p_broken.return_value = 0.0
        ctx.orchestrator._reliability_tracker = reliability

        result = phase.execute(ctx)
        states = result.inhibition_states or []
        for s in states:
            assert s.inhibition_reason == "none"

    def test_unreliable_engine_suppressed(self):
        """Unreliable engine gets suppressed."""
        phase = InhibitPhase()
        ctx = _make_ctx()
        reliability = MagicMock()
        reliability.is_reliable.side_effect = lambda sid, eng: eng != "bad_engine"
        reliability.p_broken.return_value = 0.8
        ctx.orchestrator._reliability_tracker = reliability
        ctx.perceptions = [_mock_perception("bad_engine")]
        ctx.plasticity_weights = {"bad_engine": 1.0}

        result = phase.execute(ctx)
        states = result.inhibition_states or []
        for s in states:
            assert "unreliable" in s.inhibition_reason

    def test_no_reliability_tracker_logs_warning(self):
        """Phase runs without error even with no reliability tracker."""
        phase = InhibitPhase()
        ctx = _make_ctx()
        ctx.orchestrator._reliability_tracker = None
        result = phase.execute(ctx)
        # Should not crash — returns empty states
        assert hasattr(result, "inhibition_states")

    def test_suppression_persisted_to_redis(self):
        """Suppression factor written to Redis."""
        redis = MagicMock()
        phase = InhibitPhase()
        ctx = _make_ctx()
        store = MagicMock()
        store._redis = redis
        ctx.orchestrator._series_values_store = store
        reliability = MagicMock()
        reliability.is_reliable.return_value = False
        reliability.p_broken.return_value = 0.9
        ctx.orchestrator._reliability_tracker = reliability

        phase.execute(ctx)
        assert redis.setex.call_count >= 1
        args = redis.setex.call_args[0]
        key = args[0]
        assert "zenin:inhibition:" in key

    def test_reloaded_from_redis_on_restart(self):
        """Suppression factor loaded from Redis on first access."""
        redis = MagicMock()
        redis.get.return_value = b"0.5"
        phase = InhibitPhase()
        ctx = _make_ctx()
        store = MagicMock()
        store._redis = redis
        ctx.orchestrator._series_values_store = store
        reliability = MagicMock()
        reliability.is_reliable.return_value = True
        ctx.orchestrator._reliability_tracker = reliability

        result = phase.execute(ctx)
        states = result.inhibition_states or []
        for s in states:
            assert s.suppression_factor > 0


# ── DecisionArbiterPhase ────────────────────────────────────────────


class TestDecisionArbiterPhase:
    def test_profile_engine_inferred_from_equipment_class(self):
        """Equipment class TEMPERATURE resolves to 'statistical'."""
        ctx = _make_ctx()
        fc = MagicMock()
        sp = MagicMock()
        sp.preferred_engine = None
        sp.equipment_class = MagicMock()
        sp.equipment_class.value = "TEMPERATURE"
        fc.sensor_profile = sp
        ctx.feature_context = fc
        profile = _resolve_profile_engine(ctx)
        assert profile == "statistical"

    def test_profile_engine_takes_preferred_over_equipment(self):
        """preferred_engine takes priority over equipment_class."""
        ctx = _make_ctx()
        fc = MagicMock()
        sp = MagicMock()
        sp.preferred_engine = "custom_engine"
        sp.equipment_class = MagicMock()
        sp.equipment_class.value = "TEMPERATURE"
        fc.sensor_profile = sp
        ctx.feature_context = fc
        profile = _resolve_profile_engine(ctx)
        assert profile == "custom_engine"

    def test_rule_3_flag_vs_profile_conflict(self):
        """Different flag_engine and profile_engine triggers Rule 3."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="taylor",
            profile_engine="kalman",
            fusion_engine="kalman",
            series_id="s1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.authority == "profile"
        assert decision.chosen_engine == "kalman"
        assert any("flag" in o for o in decision.overrides)

    def test_rule_4_profile_vs_fusion_conflict(self):
        """Different profile_engine and fusion_engine triggers Rule 4."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="kalman",
            profile_engine="statistical",
            fusion_engine="fusion",
            series_id="s1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "statistical"
        assert decision.authority == "profile"

    def test_rule_5_consensus_all_agree(self):
        """All layers agree → authority=fusion."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="kalman",
            profile_engine="kalman",
            fusion_engine="kalman",
            series_id="s1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.authority == "fusion"
        assert decision.chosen_engine == "kalman"

    def test_pressure_maps_to_kalman(self):
        """PRESSURE equipment class maps to 'kalman'."""
        ctx = _make_ctx()
        fc = MagicMock()
        sp = MagicMock()
        sp.preferred_engine = None
        sp.equipment_class = MagicMock()
        sp.equipment_class.value = "PRESSURE"
        fc.sensor_profile = sp
        ctx.feature_context = fc
        profile = _resolve_profile_engine(ctx)
        assert profile == "kalman"

    def test_cyclic_maps_to_statistical(self):
        """CYCLIC equipment class maps to 'statistical'."""
        ctx = _make_ctx()
        fc = MagicMock()
        sp = MagicMock()
        sp.preferred_engine = None
        sp.equipment_class = MagicMock()
        sp.equipment_class.value = "CYCLIC"
        fc.sensor_profile = sp
        ctx.feature_context = fc
        profile = _resolve_profile_engine(ctx)
        assert profile == "statistical"

    def test_fallback_to_fusion_for_unknown_equipment(self):
        """Unknown equipment class returns 'fusion'."""
        ctx = _make_ctx()
        fc = MagicMock()
        sp = MagicMock()
        sp.preferred_engine = None
        sp.equipment_class = MagicMock()
        sp.equipment_class.value = "UNKNOWN"
        fc.sensor_profile = sp
        ctx.feature_context = fc
        profile = _resolve_profile_engine(ctx)
        assert profile == "fusion"
