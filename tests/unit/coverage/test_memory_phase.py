"""Tests for MemoryPhase — async Weaviate storage and similar-case retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.memory_phase import MemoryPhase


@pytest.fixture
def mock_orchestrator():
    orchestrator = MagicMock()
    store = MagicMock()
    store._client = MagicMock()
    store.store = MagicMock()
    store.retrieve_similar = MagicMock(return_value=[])
    orchestrator._memory_registry = MagicMock()
    orchestrator._memory_registry._store = store
    return orchestrator


@pytest.fixture
def ctx(mock_orchestrator):
    from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import PipelineContext
    return PipelineContext(
        orchestrator=mock_orchestrator,
        values=[1.0, 2.0, 3.0],
        timestamps=[100.0, 200.0, 300.0],
        series_id="test_series",
        flags=None,
        timer=MagicMock(),
        fused_value=42.0,
        fused_confidence=0.85,
        fused_trend="up",
        profile=None,
        feature_context=None,
    )


class TestMemoryPhaseImportable:
    def test_importable(self):
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import memory_phase
        assert memory_phase is not None


class TestMemoryPhaseExecution:
    def test_execute_success(self, ctx):
        phase = MemoryPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_execute_missing_registry(self, ctx):
        ctx.orchestrator._memory_registry = None
        phase = MemoryPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_skip_when_no_fused_value(self, ctx):
        ctx.fused_value = None
        phase = MemoryPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_async_storage_failure(self, ctx):
        store = ctx.orchestrator._memory_registry._store
        store.store = MagicMock(side_effect=Exception("Weaviate down"))
        phase = MemoryPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_similar_cases_added_to_metadata(self, ctx):
        store = ctx.orchestrator._memory_registry._store
        store.retrieve_similar = MagicMock(return_value=[
            MagicMock(
                timestamp=1_000_000.0,
                sensor_id="eq1",
                anomaly_score=3.5,
                regime="normal",
                metadata={},
            ),
        ])
        phase = MemoryPhase()
        result = phase.execute(ctx)
        assert "similar_cases" in result.metadata
        assert len(result.metadata["similar_cases"]) >= 1

    def test_empty_similar_cases(self, ctx):
        store = ctx.orchestrator._memory_registry._store
        store.retrieve_similar = MagicMock(return_value=[])
        phase = MemoryPhase()
        result = phase.execute(ctx)
        assert "similar_cases" not in result.metadata or len(result.metadata.get("similar_cases", [])) == 0

    def test_non_blocking(self, ctx):
        import time
        t0 = time.time()
        phase = MemoryPhase()
        phase.execute(ctx)
        elapsed = time.time() - t0
        assert elapsed < 0.05

    def test_execute_without_profile(self, ctx):
        ctx.profile = None
        phase = MemoryPhase()
        result = phase.execute(ctx)
        assert result is ctx

    def test_execute_return_ctx(self, ctx):
        phase = MemoryPhase()
        result = phase.execute(ctx)
        assert result is ctx
