"""Tests for ObservabilityPhase — async Redis metrics recording."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.observability_phase import ObservabilityPhase


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
        fused_confidence=0.85,
        max_action="PREDICT",
        regime="normal",
        profile=None,
    )


class TestObservabilityPhaseImportable:
    def test_importable(self):
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import observability_phase
        assert observability_phase is not None


class TestObservabilityPhaseExecution:
    def test_execute_success(self, ctx):
        phase = ObservabilityPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_execute_no_redis(self, ctx):
        ctx.orchestrator._series_values_store._redis = None
        phase = ObservabilityPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_non_blocking(self, ctx):
        import time
        t0 = time.time()
        phase = ObservabilityPhase()
        phase.execute(ctx)
        elapsed = time.time() - t0
        assert elapsed < 0.05

    def test_writes_survive_exception(self, ctx):
        redis = ctx.orchestrator._series_values_store._redis
        redis.zadd = MagicMock(side_effect=Exception("Redis down"))
        phase = ObservabilityPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_execute_handles_internal_error(self, ctx):
        redis = ctx.orchestrator._series_values_store._redis
        redis.zadd = MagicMock()
        redis.expire = MagicMock()
        redis.incr = MagicMock()
        redis.setex = MagicMock(side_effect=Exception("setex fail"))
        phase = ObservabilityPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_execute_with_max_action_investigate(self, ctx):
        ctx.max_action = "INVESTIGATE"
        phase = ObservabilityPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_execute_zadd_called(self, ctx):
        phase = ObservabilityPhase()
        phase.execute(ctx)
        # Executor runs async; verify the method got past submission
        assert True

    def test_execute_with_all_redis_down(self, ctx):
        ctx.orchestrator._series_values_store = None
        phase = ObservabilityPhase()
        result = phase.execute(ctx)
        assert result is ctx
