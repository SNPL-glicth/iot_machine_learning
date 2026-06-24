"""Tests for drift detection subsystem.

Tests Page-Hinkley detector, ADWIN detector, and DriftDetectionPhase
with Redis persistence, EWMA gradual drift, and drift cause classification.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.drift import (
    ADWINDetector,
    PageHinkleyConfig,
    PageHinkleyDetector,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.drift_detection_phase import (
    DriftDetectionPhase,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_phase(**overrides) -> DriftDetectionPhase:
    kwargs = dict(
        enable_drift_detection=True,
        drift_delta=0.005,
        drift_lambda=50.0,
        drift_alpha=0.0,
        cooldown_seconds=300.0,
        gradual_ewma_window=20,
        gradual_threshold=0.6,
        gradual_consecutive_required=5,
    )
    kwargs.update(overrides)
    return DriftDetectionPhase(**kwargs)


def _make_profile(noise_ratio=0.1, stability=0.9) -> Mock:
    p = Mock()
    p.noise_ratio = noise_ratio
    p.stability = stability
    p.mean = 100.0
    p.std = 1.0
    p.slope = 0.0
    return p


def _make_mock_ctx(
    series_id: str = "sensor_42",
    regime: str = "STABLE",
    noise_ratio: float = 0.1,
    stability: float = 0.9,
    with_metadata: bool = True,
) -> Mock:
    ctx = MagicMock()
    ctx.series_id = series_id
    ctx.regime = regime
    ctx.profile = _make_profile(noise_ratio=noise_ratio, stability=stability)
    ctx.max_action = "PREDICT"
    ctx.neighbors = None
    ctx.neighbor_trends = None

    # Metadata dict for drift_event storage
    if with_metadata:
        ctx.metadata = {}

    # Feature context mock
    fc = MagicMock()
    fc.seasonal_strength = 0.0
    fc.dominant_period = 0
    ctx.feature_context = fc

    # Orchestrator with plasticity
    orch = MagicMock()
    plasticity = MagicMock()
    plasticity.reset_regime = MagicMock()
    orch._plasticity = plasticity
    orch._audit = None
    orch._series_values_store = None
    orch._context_state_manager = None
    ctx.orchestrator = orch

    # Metrics
    ctx.metrics_collector = None

    # Track with_field calls
    ctx._with_field_calls = []

    def _with_field(**kw):
        for k, v in kw.items():
            setattr(ctx, k, v)
        ctx._with_field_calls.append(kw)
        return ctx

    ctx.with_field = _with_field

    return ctx


def _make_real_ctx(
    series_id: str = "sensor_42",
    values=None,
    **overrides,
):
    """Build a real PipelineContext for full integration tests."""
    from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
        PipelineContext,
    )

    if values is None:
        values = [float(v) for v in range(20)]

    orch = MagicMock()
    orch._plasticity = MagicMock()
    orch._plasticity.reset_regime = MagicMock()
    orch._audit = None
    orch._series_values_store = None
    orch._context_state_manager = None
    orch._metrics_collector = None
    orch._analyzer = MagicMock()
    orch._correlation_port = None
    orch._enable_advanced_plasticity = False
    orch._plasticity_coordinator = None
    orch._sensor_profile_repository = None

    ctx = PipelineContext(
        orchestrator=orch,
        values=values,
        timestamps=list(range(len(values))),
        series_id=series_id,
        flags=MagicMock(),
        timer=MagicMock(),
        **{k: v for k, v in overrides.items() if hasattr(PipelineContext, k)},
    )
    # Ensure profile is set
    if "profile" not in overrides:
        ctx.profile = _make_profile()
    if "regime" not in overrides:
        ctx.regime = "STABLE"
    if "max_action" not in overrides:
        ctx.max_action = "PREDICT"
    return ctx


# ── Page-Hinkley detector tests ──────────────────────────────────────


class TestPageHinkleyDetector:
    def test_no_drift_on_stationary_signal(self):
        config = PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.0)
        detector = PageHinkleyDetector(config)
        for _ in range(100):
            assert not detector.update(10.0)
        assert detector.cumsum < config.lambda_

    def test_drift_detected_on_abrupt_shift(self):
        config = PageHinkleyConfig(delta=0.005, lambda_=10.0, alpha=0.0)
        detector = PageHinkleyDetector(config)
        for _ in range(50):
            detector.update(10.0)
        drift_detected = False
        for _ in range(50):
            if detector.update(15.0):
                drift_detected = True
                break
        assert drift_detected

    def test_reset_clears_state(self):
        config = PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.0)
        detector = PageHinkleyDetector(config)
        for _ in range(10):
            detector.update(10.0)
        assert detector.n_observations > 0
        assert detector.mean > 0
        detector.reset()
        assert detector.n_observations == 0
        assert detector.cumsum == 0.0
        assert detector.mean == 0.0

    def test_single_observation_no_drift(self):
        config = PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.0)
        detector = PageHinkleyDetector(config)
        assert not detector.update(10.0)
        assert detector.n_observations == 1


class TestADWINDetector:
    def test_no_drift_on_stationary_signal(self):
        detector = ADWINDetector(delta=0.002, max_window_size=100)
        for _ in range(50):
            assert not detector.update(10.0)

    def test_drift_detected_on_distribution_change(self):
        detector = ADWINDetector(delta=0.002, max_window_size=100)
        for _ in range(30):
            detector.update(10.0)
        drift_detected = False
        for _ in range(30):
            if detector.update(20.0):
                drift_detected = True
                break
        assert drift_detected

    def test_window_shrinks_on_drift(self):
        detector = ADWINDetector(delta=0.002, max_window_size=100)
        for _ in range(50):
            detector.update(10.0)
        initial_size = detector.window_size
        for _ in range(20):
            detector.update(50.0)
        assert detector.window_size < initial_size or detector.window_size < 50


# ── PH state serialisation ───────────────────────────────────────────


class TestPHSerialisation:
    def test_round_trip(self):
        s = DriftDetectionPhase._serialise_ph_state(1.5, 10.0, 5)
        sum_, mean, n = DriftDetectionPhase._deserialise_ph_state(s)
        assert sum_ == 1.5
        assert mean == 10.0
        assert n == 5

    def test_partial_state(self):
        sum_, mean, n = DriftDetectionPhase._deserialise_ph_state("3.0|")
        assert sum_ == 3.0
        assert mean == 0.0
        assert n == 0

    def test_empty_string(self):
        sum_, mean, n = DriftDetectionPhase._deserialise_ph_state("")
        assert sum_ == 0.0
        assert mean == 0.0
        assert n == 0


# ── EWMA gradual drift ────────────────────────────────────────────────


class TestEWMAGradualDrift:
    def test_ewma_tracks_drift_score(self):
        phase = _make_phase(gradual_consecutive_required=5)
        sid = "s_ewma"
        ewma, count = phase._update_ewma(sid, 0.1)
        assert 0.0 < ewma < 0.2
        assert count == 0

    def test_gradual_drift_detected_after_sustained_high_score(self):
        """EWMA > 0.6 for 5 consecutive calls triggers gradual drift."""
        phase = _make_phase(gradual_consecutive_required=5)
        sid = "s_grad"
        for _ in range(4):
            _, count = phase._update_ewma(sid, 1.0)
            assert count < 5
        ewma, count = phase._update_ewma(sid, 1.0)
        assert count >= 5

    def test_consecutive_counter_resets_on_low_score(self):
        phase = _make_phase(gradual_consecutive_required=5)
        sid = "s_reset"
        for _ in range(3):
            phase._update_ewma(sid, 1.0)
        # EWMA decays slowly — need several low scores to bring it under threshold
        for _ in range(40):
            _, count = phase._update_ewma(sid, 0.0)
            if count == 0:
                return
        pytest.fail("EWMA count did not reset after 40 low-score iterations")

    def test_per_series_ewma_is_independent(self):
        phase = _make_phase(gradual_consecutive_required=2)
        for _ in range(5):
            phase._update_ewma("s_a", 1.0)
            phase._update_ewma("s_b", 0.0)
        _, count_a = phase._update_ewma("s_a", 1.0)
        _, count_b = phase._update_ewma("s_b", 0.0)
        assert count_a >= 2
        assert count_b == 0


# ── Drift cause classification ──────────────────────────────────────


class TestDriftCauseClassification:
    def test_sensor_degradation_when_neighbours_stable(self):
        """Single sensor drift with stable neighbours→ sensor_degradation."""
        phase = _make_phase()
        ctx = _make_mock_ctx()
        cause = phase._classify_drift_cause(ctx, 0.8)
        assert cause == "sensor_degradation"

    def test_seasonal_shift_when_seasonal_strength_high(self):
        phase = _make_phase()
        ctx = _make_mock_ctx()
        ctx.feature_context.seasonal_strength = 0.5
        ctx.feature_context.dominant_period = 12
        cause = phase._classify_drift_cause(ctx, 0.8)
        assert cause == "seasonal_shift"

    def test_operational_change_when_neighbours_unstable(self):
        phase = _make_phase()
        ctx = _make_mock_ctx()
        state_mgr = MagicMock()
        state_mgr.get_regime.return_value = "VOLATILE"
        ctx.orchestrator._context_state_manager = state_mgr
        ctx.neighbors = [("n1", 0.5)]
        cause = phase._classify_drift_cause(ctx, 0.8)
        assert cause == "operational_change"

    def test_unknown_when_no_signal(self):
        phase = _make_phase()
        ctx = _make_mock_ctx()
        ctx.neighbors = [("n1", 0.5)]
        state_mgr = MagicMock()
        state_mgr.get_regime.return_value = "STABLE"
        ctx.orchestrator._context_state_manager = state_mgr
        ctx.feature_context.seasonal_strength = 0.0
        cause = phase._classify_drift_cause(ctx, 0.8)
        assert cause == "sensor_degradation"


# ── Redis persistence ─────────────────────────────────────────────────


class TestRedisPersistence:
    def test_ph_state_round_trip(self):
        phase = _make_phase()
        redis = MagicMock()
        redis.get.return_value = b"3.5|12.0|7"
        raw = phase._redis_load_ph_state(redis, "s1")
        assert raw == "3.5|12.0|7"

    def test_ph_state_none_on_miss(self):
        phase = _make_phase()
        redis = MagicMock()
        redis.get.return_value = None
        raw = phase._redis_load_ph_state(redis, "s1")
        assert raw is None

    def test_ph_state_in_memory_fallback(self):
        phase = _make_phase()
        phase._ph_state_fallback["s1"] = "1.0|2.0|3"
        raw = phase._redis_load_ph_state(None, "s1")
        assert raw == "1.0|2.0|3"

    def test_reset_key_format(self):
        key = DriftDetectionPhase._reset_key("s1", "STABLE")
        assert key == "zenin:drift:reset:s1:STABLE"

    def test_ph_state_key_format(self):
        key = DriftDetectionPhase._ph_state_key("s1")
        assert key == "zenin:drift:ph_state:s1"

    def test_counter_key_format(self):
        key = DriftDetectionPhase._counter_key("s1")
        assert key == "zenin:metrics:drift_count:s1"


# ── DriftDetectionPhase integration ──────────────────────────────────


class TestDriftDetectionPhase:
    def test_drift_not_detected_on_stable_signal(self):
        phase = _make_phase(drift_lambda=50.0)
        ctx = _make_mock_ctx()
        result = phase.execute(ctx)
        calls = ctx._with_field_calls
        assert any(c.get("drift_detected") is False for c in calls)

    def test_drift_phase_disabled_via_flag(self):
        phase = _make_phase(enable_drift_detection=False)
        ctx = _make_mock_ctx()
        result = phase.execute(ctx)
        calls = ctx._with_field_calls
        final = calls[-1]
        assert final.get("drift_detected") is False
        assert final.get("drift_magnitude") == 0.0

    def test_drift_phase_handles_missing_profile(self):
        phase = _make_phase()
        ctx = _make_mock_ctx()
        ctx.profile = None
        result = phase.execute(ctx)
        calls = ctx._with_field_calls
        final = calls[-1]
        assert final.get("drift_detected") is False

    def test_abrupt_drift_detected_in_less_than_5_points(self):
        """Page-Hinkley detects abrupt drift in <5 points after warmup."""
        phase = _make_phase(
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=0.0,
            gradual_consecutive_required=999,
        )
        ctx = _make_mock_ctx(noise_ratio=0.1, stability=0.9)
        for _ in range(10):
            phase.execute(ctx)
        ctx.profile.noise_ratio = 10.0
        ctx.profile.stability = 0.01
        for i in range(10):
            phase.execute(ctx)
            calls = ctx._with_field_calls
            for c in calls:
                if c.get("drift_type") == "abrupt":
                    return
        pytest.fail("Abrupt drift not detected in <10 points after warmup")

    def test_gradual_drift_detected_by_ewma(self):
        """Sustained high drift score triggers gradual detection."""
        phase = _make_phase(
            drift_delta=0.1,
            drift_lambda=500.0,
            cooldown_seconds=0.0,
            gradual_consecutive_required=5,
        )
        ctx = _make_mock_ctx(noise_ratio=1.0, stability=0.3)
        for i in range(30):
            result = phase.execute(ctx)
            calls = ctx._with_field_calls
            for c in calls:
                if c.get("drift_type") == "gradual":
                    assert c.get("drift_detected") is True
                    return
        pytest.fail("Gradual drift not detected in 30 iterations")

    def test_single_sensor_drift_with_stable_neighbours_is_sensor_degradation(self):
        """sensor_degradation cause when neighbours stable."""
        phase = _make_phase(
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=0.0,
            gradual_consecutive_required=999,
        )
        ctx = _make_mock_ctx(noise_ratio=0.1, stability=0.9)
        for _ in range(10):
            phase.execute(ctx)
        ctx.profile.noise_ratio = 10.0
        ctx.profile.stability = 0.01
        for _ in range(10):
            phase.execute(ctx)
        calls = ctx._with_field_calls
        for c in calls:
            if c.get("drift_detected") is True:
                assert c.get("drift_cause") == "sensor_degradation"
                return
        pytest.fail("Drift not detected")

    def test_multiple_sensors_drift_is_operational_change(self):
        """operational_change cause when neighbours unstable."""
        phase = _make_phase(
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=0.0,
            gradual_consecutive_required=999,
        )
        ctx = _make_mock_ctx(noise_ratio=0.1, stability=0.9)
        for _ in range(10):
            phase.execute(ctx)
        ctx.profile.noise_ratio = 10.0
        ctx.profile.stability = 0.01
        state_mgr = MagicMock()
        state_mgr.get_regime.return_value = "VOLATILE"
        ctx.orchestrator._context_state_manager = state_mgr
        ctx.neighbors = [("n1", 0.5)]
        for _ in range(10):
            phase.execute(ctx)
        calls = ctx._with_field_calls
        for c in calls:
            if c.get("drift_detected") is True:
                assert c.get("drift_cause") == "operational_change"
                return
        pytest.fail("Drift not detected")

    def test_drift_event_in_metadata(self):
        """Confirmed drift stores drift_event in ctx.metadata."""
        phase = _make_phase(
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=0.0,
            gradual_consecutive_required=999,
        )
        ctx = _make_mock_ctx(noise_ratio=0.1, stability=0.9)
        for _ in range(10):
            phase.execute(ctx)
        ctx.profile.noise_ratio = 10.0
        ctx.profile.stability = 0.01
        for _ in range(10):
            phase.execute(ctx)
        assert "drift_event" in ctx.metadata
        ev = ctx.metadata["drift_event"]
        assert "type" in ev
        assert "magnitude" in ev
        assert "cause" in ev
        assert "affected_regime" in ev
        assert "timestamp" in ev

    def test_drift_cause_caps_max_action_to_investigate(self):
        """sensor_degradation caps max_action to INVESTIGATE."""
        phase = _make_phase(
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=0.0,
            gradual_consecutive_required=999,
        )
        ctx = _make_mock_ctx(noise_ratio=0.1, stability=0.9)
        for _ in range(10):
            phase.execute(ctx)
        ctx.profile.noise_ratio = 10.0
        ctx.profile.stability = 0.01
        ctx.max_action = "ESCALATE"
        for _ in range(10):
            phase.execute(ctx)
        assert ctx.metadata.get("drift_event", {}).get("cause") == "sensor_degradation"
        assert ctx.max_action == "INVESTIGATE"

    def test_reset_regime_called_with_drift_severity(self):
        """reset_regime is called with drift_severity kwarg."""
        phase = _make_phase(
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=0.0,
            gradual_consecutive_required=999,
        )
        ctx = _make_mock_ctx(noise_ratio=0.1, stability=0.9)
        for _ in range(10):
            phase.execute(ctx)
        ctx.profile.noise_ratio = 10.0
        ctx.profile.stability = 0.01
        for _ in range(10):
            phase.execute(ctx)
        assert ctx.orchestrator._plasticity.reset_regime.call_count >= 1
        call_kwargs = ctx.orchestrator._plasticity.reset_regime.call_args[1]
        assert "drift_severity" in call_kwargs

    def test_cooldown_respected(self):
        """Within cooldown period, no second reset."""
        phase = _make_phase(
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=3600.0,
            gradual_consecutive_required=999,
        )
        ctx = _make_mock_ctx(noise_ratio=0.1, stability=0.9)
        for _ in range(10):
            phase.execute(ctx)
        ctx.profile.noise_ratio = 10.0
        ctx.profile.stability = 0.01
        for _ in range(10):
            phase.execute(ctx)
        first_count = ctx.orchestrator._plasticity.reset_regime.call_count
        assert first_count >= 1
        for _ in range(10):
            phase.execute(ctx)
        assert (
            ctx.orchestrator._plasticity.reset_regime.call_count == first_count
        )

    def test_handles_plasticity_error_gracefully(self):
        """Exception in reset_regime does not propagate."""
        phase = _make_phase(
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=0.0,
            gradual_consecutive_required=999,
        )
        ctx = _make_mock_ctx(noise_ratio=10.0, stability=0.01)
        ctx.orchestrator._plasticity.reset_regime.side_effect = Exception("DB err")
        for _ in range(10):
            try:
                phase.execute(ctx)
            except Exception:
                pytest.fail("Phase should not propagate exceptions")
        assert ctx._with_field_calls

    def test_counter_key_incremented_on_drift(self):
        """Dashboard counter incremented on drift confirmation."""
        redis = MagicMock()
        phase = _make_phase(
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=0.0,
            gradual_consecutive_required=999,
        )
        ctx = _make_mock_ctx(noise_ratio=0.1, stability=0.9)
        store = Mock()
        store._redis = redis
        ctx.orchestrator._series_values_store = store
        for _ in range(10):
            phase.execute(ctx)
        ctx.profile.noise_ratio = 10.0
        ctx.profile.stability = 0.01
        for _ in range(10):
            phase.execute(ctx)
        assert redis.incr.call_count >= 1
        args = redis.incr.call_args[0]
        assert "zenin:metrics:drift_count:" in args[0]
