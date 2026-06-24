"""Tests for ActionGuardPhase — real-time series_state from Redis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.action_guard_phase import ActionGuardPhase
from iot_machine_learning.domain.services.action_guard import GuardedAction


@pytest.fixture
def mock_orch_with_redis():
    orchestrator = MagicMock()
    orchestrator._series_values_store = MagicMock()
    orchestrator._series_values_store._redis = MagicMock()
    return orchestrator


@pytest.fixture
def ctx(mock_orch_with_redis):
    from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import PipelineContext
    return PipelineContext(
        orchestrator=mock_orch_with_redis,
        values=[1.0, 2.0, 3.0],
        timestamps=[100.0, 200.0, 300.0],
        series_id="test_series",
        flags=None,
        timer=MagicMock(),
        max_action="INVESTIGATE",
    )


class TestActionGuardPhaseImportable:
    def test_importable(self):
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import action_guard_phase
        assert action_guard_phase is not None


class TestActionGuardPhaseExecution:
    def test_execute_active_state(self, ctx):
        redis = ctx.orchestrator._series_values_store._redis
        redis.get = MagicMock(side_effect=lambda k: b"100.0" if "zenin:series_state" in str(k) else b"15")
        redis.setex = MagicMock()
        redis.incr = MagicMock()
        redis.expire = MagicMock()

        with patch("iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.action_guard_phase.time.time", return_value=200.0):
            phase = ActionGuardPhase()
            result = phase.execute(ctx)

        assert result is ctx
        assert result.guarded_action is not None
        assert result.guarded_action.series_state == "ACTIVE"

    def test_execute_initializing(self, ctx):
        redis = ctx.orchestrator._series_values_store._redis
        redis.get = MagicMock(side_effect=lambda k: b"100.0" if "zenin:series_state" in str(k) else b"3")
        redis.setex = MagicMock()
        redis.incr = MagicMock()
        redis.expire = MagicMock()

        with patch("iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.action_guard_phase.time.time", return_value=200.0):
            phase = ActionGuardPhase()
            result = phase.execute(ctx)

        assert result.guarded_action.series_state == "INITIALIZING"
        assert result.guarded_action.action_allowed is False

    def test_execute_offline(self, ctx):
        redis = ctx.orchestrator._series_values_store._redis
        redis.get = MagicMock(side_effect=lambda k: b"100.0" if "zenin:series_state" in str(k) else b"15")
        redis.setex = MagicMock()
        redis.incr = MagicMock()
        redis.expire = MagicMock()

        with patch("iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.action_guard_phase.time.time", return_value=1_000_000.0):
            phase = ActionGuardPhase()
            result = phase.execute(ctx)

        assert result.guarded_action.series_state == "OFFLINE"
        assert result.guarded_action.action_allowed is False

    def test_execute_stale(self, ctx):
        redis = ctx.orchestrator._series_values_store._redis
        redis.get = MagicMock(side_effect=lambda k: b"100.0" if "zenin:series_state" in str(k) else b"15")
        redis.setex = MagicMock()
        redis.incr = MagicMock()
        redis.expire = MagicMock()

        with patch("iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.action_guard_phase.time.time", return_value=50_000.0):
            phase = ActionGuardPhase()
            result = phase.execute(ctx)

        assert result.guarded_action.series_state == "STALE"

    def test_execute_no_redis(self, ctx):
        ctx.orchestrator._series_values_store._redis = None

        phase = ActionGuardPhase()
        result = phase.execute(ctx)

        assert result is ctx

    def test_execute_redis_failure(self, ctx):
        redis = ctx.orchestrator._series_values_store._redis
        redis.get = MagicMock(side_effect=Exception("Redis down"))

        phase = ActionGuardPhase()
        result = phase.execute(ctx)

        assert result is ctx

    def test_predict_action_passes_through(self, ctx):
        ctx.max_action = "PREDICT"
        redis = ctx.orchestrator._series_values_store._redis
        redis.get = MagicMock(side_effect=lambda k: b"100.0" if "zenin:series_state" in str(k) else b"15")
        redis.setex = MagicMock()
        redis.incr = MagicMock()
        redis.expire = MagicMock()

        with patch("iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.action_guard_phase.time.time", return_value=200.0):
            phase = ActionGuardPhase()
            result = phase.execute(ctx)

        assert result.guarded_action is not None

    def test_series_state_persisted(self, ctx):
        redis = ctx.orchestrator._series_values_store._redis
        redis.get = MagicMock(return_value=b"100.0")
        redis.setex = MagicMock()
        redis.incr = MagicMock()
        redis.expire = MagicMock()

        with patch("iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.action_guard_phase.time.time", return_value=200.0):
            phase = ActionGuardPhase()
            phase.execute(ctx)

        redis.setex.assert_called_once()
        redis.incr.assert_called_once()

    def test_unexpected_error_returns_ctx(self, ctx):
        redis = ctx.orchestrator._series_values_store._redis
        redis.get = MagicMock(return_value=b"100.0")
        redis.setex = MagicMock()
        redis.incr = MagicMock()
        redis.expire = MagicMock()

        with patch("iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.action_guard_phase.ActionGuard.guard", side_effect=Exception("guard error")):
            phase = ActionGuardPhase()
            result = phase.execute(ctx)

        assert result is ctx
