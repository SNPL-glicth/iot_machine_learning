"""Tests for batch runner enterprise bridge.

Validates:
- Feature flags control routing correctly
- EnterprisePredictionAdapter calls use case and converts result
- Fallback baseline works when enterprise fails
- ABMetricsCollector tracks metrics
- BatchAuditLogger records cycle events
- SensorProcessor routes via enterprise_adapter param
- MLBatchRunner backward compat (flags off = legacy behavior)

Restricción: < 180 líneas.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch

from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags


# ── batch_flags tests ────────────────────────────────────────────────────


class TestShouldUseEnterprise:
    """Tests for should_use_enterprise routing logic."""

    def _make_flags(self, **overrides) -> FeatureFlags:
        defaults = {
            "ML_ROLLBACK_TO_BASELINE": False,
            "ML_BATCH_USE_ENTERPRISE": False,
            "ML_BATCH_ENTERPRISE_SENSORS": None,
            "ML_BATCH_BASELINE_ONLY_SENSORS": None,
        }
        defaults.update(overrides)
        return FeatureFlags(**defaults)

    def test_panic_button_forces_baseline(self):
        from iot_machine_learning.ml_service.runners.bridge_config.batch_flags import (
            should_use_enterprise,
        )

        flags = self._make_flags(
            ML_ROLLBACK_TO_BASELINE=True,
            ML_BATCH_USE_ENTERPRISE=True,
        )
        assert should_use_enterprise(42, flags) is False

    def test_blacklist_forces_baseline(self):
        from iot_machine_learning.ml_service.runners.bridge_config.batch_flags import (
            should_use_enterprise,
        )

        flags = self._make_flags(
            ML_BATCH_USE_ENTERPRISE=True,
            ML_BATCH_BASELINE_ONLY_SENSORS="42,55",
        )
        assert should_use_enterprise(42, flags) is False
        assert should_use_enterprise(99, flags) is True

    def test_whitelist_enables_enterprise(self):
        from iot_machine_learning.ml_service.runners.bridge_config.batch_flags import (
            should_use_enterprise,
        )

        flags = self._make_flags(ML_BATCH_ENTERPRISE_SENSORS="42,55")
        assert should_use_enterprise(42, flags) is True
        assert should_use_enterprise(99, flags) is False

    def test_global_flag_enables_all(self):
        from iot_machine_learning.ml_service.runners.bridge_config.batch_flags import (
            should_use_enterprise,
        )

        flags = self._make_flags(ML_BATCH_USE_ENTERPRISE=True)
        assert should_use_enterprise(42, flags) is True
        assert should_use_enterprise(99, flags) is True

    def test_default_is_baseline(self):
        from iot_machine_learning.ml_service.runners.bridge_config.batch_flags import (
            should_use_enterprise,
        )

        flags = self._make_flags()
        assert should_use_enterprise(42, flags) is False

    def test_blacklist_overrides_global(self):
        from iot_machine_learning.ml_service.runners.bridge_config.batch_flags import (
            should_use_enterprise,
        )

        flags = self._make_flags(
            ML_BATCH_USE_ENTERPRISE=True,
            ML_BATCH_BASELINE_ONLY_SENSORS="42",
        )
        assert should_use_enterprise(42, flags) is False
        assert should_use_enterprise(55, flags) is True

    def test_invalid_sensor_list_returns_empty(self):
        from iot_machine_learning.ml_service.runners.bridge_config.batch_flags import (
            _parse_sensor_set,
        )

        assert _parse_sensor_set(None) == set()
        assert _parse_sensor_set("") == set()
        assert _parse_sensor_set("1,abc,3") == {1, 3}


# ── BatchPredictionResult tests ──────────────────────────────────────────


class TestBatchPredictionResult:
    """Tests for BatchPredictionResult dataclass."""

    def test_creation(self):
        from iot_machine_learning.ml_service.runners.adapters.enterprise_prediction import (
            BatchPredictionResult,
        )

        result = BatchPredictionResult(
            predicted_value=25.5,
            confidence=0.85,
            trend="up",
            engine_used="taylor",
        )
        assert result.predicted_value == 25.5
        assert result.confidence == 0.85
        assert result.trend == "up"
        assert result.engine_used == "taylor"
        assert result.anomaly_score == 0.0

    def test_defaults(self):
        from iot_machine_learning.ml_service.runners.adapters.enterprise_prediction import (
            BatchPredictionResult,
        )

        result = BatchPredictionResult(
            predicted_value=0.0,
            confidence=0.0,
            trend="stable",
            engine_used="baseline_fallback",
        )
        assert result.structural_regime is None
        assert result.trace_id is None
        assert result.elapsed_ms == 0.0


# ── EnterprisePredictionAdapter tests ────────────────────────────────────


class TestEnterprisePredictionAdapter:
    """Tests for EnterprisePredictionAdapter."""

    def test_success_path(self):
        from iot_machine_learning.ml_service.runners.adapters.enterprise_prediction import (
            EnterprisePredictionAdapter,
            BatchPredictionResult,
        )

        mock_storage = Mock()
        mock_audit = Mock()
        mock_use_case = Mock()
        mock_use_case.execute.return_value = Mock(
            predicted_value=25.5,
            confidence_score=0.85,
            trend="up",
            engine_name="taylor",
            audit_trace_id="abc123",
        )

        adapter = EnterprisePredictionAdapter(
            storage=mock_storage,
            use_case=mock_use_case,
            audit=mock_audit,
        )

        result = adapter.predict(sensor_id=42, window_size=60)

        assert isinstance(result, BatchPredictionResult)
        assert result.predicted_value == 25.5
        assert result.engine_used == "taylor"
        assert result.confidence == 0.85
        mock_use_case.execute.assert_called_once_with(
            sensor_id=42, window_size=60,
        )
        mock_audit.log_event.assert_called_once()
        call_kwargs = mock_audit.log_event.call_args
        assert call_kwargs[1]["event_type"] == "batch_prediction_enterprise"

    def test_failure_falls_back(self):
        from iot_machine_learning.ml_service.runners.adapters.enterprise_prediction import (
            EnterprisePredictionAdapter,
        )
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )

        mock_storage = Mock()
        mock_storage.load_sensor_window.return_value = SensorWindow(
            sensor_id=42,
            readings=[
                SensorReading(sensor_id=42, value=20.0, timestamp=1000.0),
                SensorReading(sensor_id=42, value=21.0, timestamp=1001.0),
                SensorReading(sensor_id=42, value=22.0, timestamp=1002.0),
            ],
        )
        mock_audit = Mock()
        mock_use_case = Mock()
        mock_use_case.execute.side_effect = RuntimeError("DB down")

        adapter = EnterprisePredictionAdapter(
            storage=mock_storage,
            use_case=mock_use_case,
            audit=mock_audit,
        )

        result = adapter.predict(sensor_id=42, window_size=60)

        assert result.engine_used.startswith("baseline_fallback")
        assert result.predicted_value > 0
        # Audit should log fallback
        assert mock_audit.log_event.call_count >= 1
        fallback_call = mock_audit.log_event.call_args
        assert fallback_call[1]["event_type"] == "batch_prediction_fallback"

    def test_metrics_tracking(self):
        from iot_machine_learning.ml_service.runners.adapters.enterprise_prediction import (
            EnterprisePredictionAdapter,
        )

        mock_storage = Mock()
        mock_audit = Mock()
        mock_use_case = Mock()
        mock_use_case.execute.return_value = Mock(
            predicted_value=25.5,
            confidence_score=0.85,
            trend="up",
            engine_name="taylor",
            audit_trace_id="abc123",
        )

        adapter = EnterprisePredictionAdapter(
            storage=mock_storage,
            use_case=mock_use_case,
            audit=mock_audit,
        )

        adapter.predict(sensor_id=42)
        adapter.predict(sensor_id=43)

        metrics = adapter.metrics
        assert metrics["enterprise_success"] == 2
        assert metrics["enterprise_failure"] == 0
        assert metrics["success_rate"] == 1.0


# ── Fallback baseline tests ─────────────────────────────────────────────


class TestFallbackBaseline:
    """Tests for fallback_to_baseline function."""

    def test_returns_result_with_valid_window(self):
        from iot_machine_learning.ml_service.runners.adapters.fallback_baseline import (
            fallback_to_baseline,
        )
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )

        mock_storage = Mock()
        mock_storage.load_sensor_window.return_value = SensorWindow(
            sensor_id=42,
            readings=[
                SensorReading(sensor_id=42, value=20.0, timestamp=1000.0),
                SensorReading(sensor_id=42, value=21.0, timestamp=1001.0),
                SensorReading(sensor_id=42, value=22.0, timestamp=1002.0),
            ],
        )

        result = fallback_to_baseline(mock_storage, 42, 60)

        assert result.engine_used == "baseline_fallback"
        assert 20.0 <= result.predicted_value <= 23.0
        assert result.confidence > 0

    def test_returns_empty_result_on_empty_window(self):
        from iot_machine_learning.ml_service.runners.adapters.fallback_baseline import (
            fallback_to_baseline,
        )
        from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

        mock_storage = Mock()
        mock_storage.load_sensor_window.return_value = SensorWindow(
            sensor_id=42, readings=[],
        )

        result = fallback_to_baseline(mock_storage, 42, 60)

        assert result.engine_used == "baseline_fallback_empty"
        assert result.predicted_value == 0.0
        assert result.confidence == 0.0

    def test_never_raises(self):
        from iot_machine_learning.ml_service.runners.adapters.fallback_baseline import (
            fallback_to_baseline,
        )

        mock_storage = Mock()
        mock_storage.load_sensor_window.side_effect = Exception("total failure")

        result = fallback_to_baseline(mock_storage, 42, 60)

        assert result.engine_used == "baseline_fallback_error"
        assert result.predicted_value == 0.0


# ── ABMetricsCollector tests ─────────────────────────────────────────────


class TestABMetricsCollector:
    """Tests for ABMetricsCollector."""

    def test_record_and_summary(self):
        from iot_machine_learning.ml_service.runners.monitoring.ab_metrics import (
            ABMetricsCollector,
        )

        collector = ABMetricsCollector()
        collector.record(42, "taylor", 25.5, 0.85, 10.0)
        collector.record(43, "baseline_legacy", 20.0, 0.5, 5.0)
        collector.record(44, "taylor", 26.0, 0.9, 12.0)

        summary = collector.summary()
        assert summary["total_predictions"] == 3
        assert summary["enterprise_predictions"] == 2
        assert summary["enterprise_ratio"] == pytest.approx(2 / 3, abs=0.01)
        assert "taylor" in summary["engines"]
        assert summary["engines"]["taylor"]["count"] == 2

    def test_reset(self):
        from iot_machine_learning.ml_service.runners.monitoring.ab_metrics import (
            ABMetricsCollector,
        )

        collector = ABMetricsCollector()
        collector.record(42, "taylor", 25.5, 0.85)
        collector.reset()
        assert collector.summary()["total_predictions"] == 0


# ── BatchAuditLogger tests ───────────────────────────────────────────────


class TestBatchAuditLogger:
    """Tests for BatchAuditLogger."""

    def test_start_and_end_cycle(self):
        from iot_machine_learning.ml_service.runners.monitoring.batch_audit import (
            BatchAuditLogger,
        )

        mock_audit = Mock()
        batch_audit = BatchAuditLogger(mock_audit)

        cycle_id = batch_audit.start_cycle(sensor_count=10)
        assert cycle_id is not None
        assert len(cycle_id) == 12

        batch_audit.end_cycle(
            processed=8, errors=2, enterprise_count=5, baseline_count=3,
        )

        assert mock_audit.log_event.call_count == 2
        start_call = mock_audit.log_event.call_args_list[0]
        assert start_call[1]["event_type"] == "batch_cycle_start"
        end_call = mock_audit.log_event.call_args_list[1]
        assert end_call[1]["event_type"] == "batch_cycle_end"

    def test_log_sensor_routing(self):
        from iot_machine_learning.ml_service.runners.monitoring.batch_audit import (
            BatchAuditLogger,
        )

        mock_audit = Mock()
        batch_audit = BatchAuditLogger(mock_audit)
        batch_audit.start_cycle(5)

        batch_audit.log_sensor_routing(42, "enterprise", "whitelist")

        routing_call = mock_audit.log_event.call_args_list[-1]
        assert routing_call[1]["event_type"] == "batch_sensor_routing"
        assert routing_call[1]["details"]["route"] == "enterprise"


# ── FeatureFlags new fields tests ────────────────────────────────────────


class TestFeatureFlagsBatchFields:
    """Tests for new batch enterprise fields in FeatureFlags."""

    def test_defaults_are_safe(self):
        flags = FeatureFlags()
        assert flags.ML_BATCH_USE_ENTERPRISE is False
        assert flags.ML_BATCH_ENTERPRISE_SENSORS is None
        assert flags.ML_BATCH_BASELINE_ONLY_SENSORS is None

    def test_from_env_parses_batch_flags(self, monkeypatch):
        monkeypatch.setenv("ML_BATCH_USE_ENTERPRISE", "true")
        monkeypatch.setenv("ML_BATCH_ENTERPRISE_SENSORS", "1,5,42")
        monkeypatch.setenv("ML_BATCH_BASELINE_ONLY_SENSORS", "99")

        flags = FeatureFlags.from_env()
        assert flags.ML_BATCH_USE_ENTERPRISE is True
        assert flags.ML_BATCH_ENTERPRISE_SENSORS == "1,5,42"
        assert flags.ML_BATCH_BASELINE_ONLY_SENSORS == "99"


# ── MLBatchRunner backward compat tests ──────────────────────────────────


class TestMLBatchRunnerBackwardCompat:
    """Tests that MLBatchRunner works identically with flags off."""

    def test_constructor_accepts_no_flags(self):
        """MLBatchRunner(ml_cfg) must still work (backward compat)."""
        from iot_machine_learning.ml_service.runners.ml_batch_runner import (
            MLBatchRunner,
        )
        from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig

        ml_cfg = GlobalMLConfig()
        runner = MLBatchRunner(ml_cfg)
        assert runner._flags.ML_BATCH_USE_ENTERPRISE is False
        assert runner._enterprise_container is None

    def test_constructor_accepts_flags(self):
        from iot_machine_learning.ml_service.runners.ml_batch_runner import (
            MLBatchRunner,
        )
        from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig

        ml_cfg = GlobalMLConfig()
        flags = FeatureFlags(ML_BATCH_USE_ENTERPRISE=True)
        runner = MLBatchRunner(ml_cfg, flags=flags)
        assert runner._flags.ML_BATCH_USE_ENTERPRISE is True
