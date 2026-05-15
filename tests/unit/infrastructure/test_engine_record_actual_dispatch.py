"""Tests for engine record_actual propagation via record_actual_dispatch.

Fix 2: Verifies that record_actual_dispatch forwards actual values
to individual engines for online learning.
"""

from unittest.mock import MagicMock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.perception.record_actual_handler import (
    record_actual_dispatch,
    record_actual_legacy,
)


class MockPerception:
    def __init__(self, engine_name: str, predicted_value: float):
        self.engine_name = engine_name
        self.predicted_value = predicted_value


class TestEngineRecordActualPropagation:
    """Test 1: Statistical receives record_actual."""

    def test_statistical_receives_record_actual(self) -> None:
        """record_actual_dispatch propagates to Statistical engine."""
        from iot_machine_learning.infrastructure.ml.engines.statistical import (
            StatisticalPredictionEngine,
        )

        statistical_engine = StatisticalPredictionEngine(
            enable_optimization=True,
        )
        perception = MockPerception(engine_name="statistical_ema_holt", predicted_value=10.0)

        record_actual_dispatch(
            actual_value=12.0,
            last_regime="stable",
            last_perceptions=[perception],
            last_signal_context=None,
            enable_advanced_plasticity=False,
            plasticity_coordinator=None,
            plasticity_tracker=None,
            error_history=None,
            storage=None,
            series_id="test-series",
            series_context=None,
            engines=[statistical_engine],
        )

        assert statistical_engine._prediction_count == 1

    def test_engines_none_is_backward_compatible(self) -> None:
        """record_actual_dispatch without engines does not raise."""
        perception = MockPerception(engine_name="kalman", predicted_value=5.0)

        record_actual_dispatch(
            actual_value=5.5,
            last_regime="stable",
            last_perceptions=[perception],
            last_signal_context=None,
            enable_advanced_plasticity=False,
            plasticity_coordinator=None,
            plasticity_tracker=None,
            error_history=None,
            storage=None,
            series_id="test-series",
            series_context=None,
        )

    def test_engine_record_actual_exception_is_silenced(self) -> None:
        """Exception in engine.record_actual does not break plasticity."""
        bad_engine = MagicMock()
        bad_engine.name = "bad_engine"
        bad_engine.record_actual.side_effect = RuntimeError("boom")

        perception = MockPerception(engine_name="bad_engine", predicted_value=7.0)

        # Should not raise
        record_actual_dispatch(
            actual_value=8.0,
            last_regime="stable",
            last_perceptions=[perception],
            last_signal_context=None,
            enable_advanced_plasticity=False,
            plasticity_coordinator=None,
            plasticity_tracker=None,
            error_history=None,
            storage=None,
            series_id="test-series",
            series_context=None,
            engines=[bad_engine],
        )

    def test_perceptions_without_matching_engine_are_skipped(self) -> None:
        """Perceptions for engines not in the list are silently skipped."""
        engine = MagicMock()
        engine.name = "taylor"

        perception = MockPerception(engine_name="kalman", predicted_value=3.0)

        record_actual_dispatch(
            actual_value=3.5,
            last_regime="stable",
            last_perceptions=[perception],
            last_signal_context=None,
            enable_advanced_plasticity=False,
            plasticity_coordinator=None,
            plasticity_tracker=None,
            error_history=None,
            storage=None,
            series_id="test-series",
            series_context=None,
            engines=[engine],
        )

        engine.record_actual.assert_not_called()
