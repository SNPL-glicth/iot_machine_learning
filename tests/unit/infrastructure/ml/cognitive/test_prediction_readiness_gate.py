"""Tests for PredictionReadinessGate.

Covers:
  * score >= 0.5 → max_action = "PREDICT", no fallback
  * 0.3 <= score < 0.5 → max_action = "INVESTIGATE", no fallback
  * score < 0.3 → max_action = "LOG_ONLY", is_fallback=True
  * Default score (1.0) when no data_quality_score set
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.prediction_readiness_gate import (
    PredictionReadinessGate,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
    create_initial_context,
)


def _make_ctx(data_quality_score: float = 1.0):
    ctx = create_initial_context(
        orchestrator=MagicMock(),
        values=[1.0, 2.0, 3.0],
        timestamps=None,
        series_id="test-series",
        flags=MagicMock(),
        timer=MagicMock(),
    )
    ctx.data_quality_score = data_quality_score
    return ctx


class TestPredictionReadinessGate:
    def test_high_score_allows_predict(self) -> None:
        gate = PredictionReadinessGate()
        ctx = _make_ctx(data_quality_score=0.8)
        result = gate.execute(ctx)

        assert result.max_action == "PREDICT"
        assert not result.is_fallback

    def test_score_at_threshold_investigate_allows_predict(self) -> None:
        gate = PredictionReadinessGate()
        ctx = _make_ctx(data_quality_score=0.5)
        result = gate.execute(ctx)

        assert result.max_action == "PREDICT"
        assert not result.is_fallback

    def test_score_below_investigate_threshold_caps_to_investigate(self) -> None:
        gate = PredictionReadinessGate()
        ctx = _make_ctx(data_quality_score=0.4)
        result = gate.execute(ctx)

        assert result.max_action == "INVESTIGATE"
        assert not result.is_fallback

    def test_score_at_investigate_threshold_floor_still_investigate(self) -> None:
        gate = PredictionReadinessGate()
        ctx = _make_ctx(data_quality_score=0.3)
        result = gate.execute(ctx)

        assert result.max_action == "INVESTIGATE"
        assert not result.is_fallback

    def test_score_below_log_only_threshold_triggers_fallback(self) -> None:
        gate = PredictionReadinessGate()
        ctx = _make_ctx(data_quality_score=0.2)
        result = gate.execute(ctx)

        assert result.max_action == "LOG_ONLY"
        assert result.is_fallback is True
        assert result.fallback_reason == "quality_too_low_log_only"

    def test_score_zero_triggers_log_only(self) -> None:
        gate = PredictionReadinessGate()
        ctx = _make_ctx(data_quality_score=0.0)
        result = gate.execute(ctx)

        assert result.max_action == "LOG_ONLY"
        assert result.is_fallback is True

    def test_default_score_when_not_set(self) -> None:
        gate = PredictionReadinessGate()
        ctx = create_initial_context(
            orchestrator=MagicMock(),
            values=[1.0, 2.0, 3.0],
            timestamps=None,
            series_id="test-series",
            flags=MagicMock(),
            timer=MagicMock(),
        )
        result = gate.execute(ctx)

        # Default data_quality_score is 1.0
        assert result.max_action == "PREDICT"
        assert not result.is_fallback

    def test_negative_score_triggers_log_only(self) -> None:
        gate = PredictionReadinessGate()
        ctx = _make_ctx(data_quality_score=-0.1)
        result = gate.execute(ctx)

        assert result.max_action == "LOG_ONLY"
        assert result.is_fallback is True
