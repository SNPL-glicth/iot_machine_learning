"""Unit tests for CognitiveStorageDecorator.

All dependencies are mocked — no SQL Server or Weaviate required.
Tests verify:
    - Decorator delegates all reads to inner StoragePort
    - save_prediction: SQL first, then cognitive remember_explanation
    - save_anomaly_event: SQL first, then cognitive remember_anomaly
    - Cognitive failure does NOT propagate to caller
    - Flag off → no cognitive call at all
    - Async mode fires in background thread
    - SQL failure → no cognitive call
    - Decorator isinstance check (is a StoragePort)
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

from iot_machine_learning.domain.entities.anomaly import (
    AnomalyResult,
    AnomalySeverity,
)
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.domain.ports.cognitive_memory_port import (
    CognitiveMemoryPort,
)
from iot_machine_learning.domain.ports.storage_port import StoragePort
from iot_machine_learning.infrastructure.adapters.cognitive_storage_decorator import (
    CognitiveStorageDecorator,
)
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_inner() -> MagicMock:
    """Mock inner StoragePort."""
    mock = MagicMock(spec=StoragePort)
    mock.save_prediction.return_value = 42
    mock.save_anomaly_event.return_value = 99
    mock.load_sensor_window.return_value = SensorWindow(
        sensor_id=1, readings=[]
    )
    mock.list_active_sensor_ids.return_value = [1, 2, 3]
    mock.get_latest_prediction.return_value = None
    mock.get_sensor_metadata.return_value = {"sensor_id": 1}
    mock.get_device_id_for_sensor.return_value = 7
    return mock


@pytest.fixture
def mock_cognitive() -> MagicMock:
    """Mock CognitiveMemoryPort."""
    mock = MagicMock(spec=CognitiveMemoryPort)
    mock.remember_explanation.return_value = "uuid-expl-1"
    mock.remember_anomaly.return_value = "uuid-anom-1"
    return mock


@pytest.fixture
def flags_enabled() -> FeatureFlags:
    """Flags with cognitive memory enabled, sync mode."""
    return FeatureFlags(
        ML_ENABLE_COGNITIVE_MEMORY=True,
        ML_COGNITIVE_MEMORY_ASYNC=False,
        ML_COGNITIVE_MEMORY_DRY_RUN=False,
        ML_COGNITIVE_MEMORY_URL="http://localhost:8080",
    )


@pytest.fixture
def flags_disabled() -> FeatureFlags:
    """Flags with cognitive memory disabled."""
    return FeatureFlags(
        ML_ENABLE_COGNITIVE_MEMORY=False,
    )


@pytest.fixture
def flags_async() -> FeatureFlags:
    """Flags with cognitive memory enabled, async mode."""
    return FeatureFlags(
        ML_ENABLE_COGNITIVE_MEMORY=True,
        ML_COGNITIVE_MEMORY_ASYNC=True,
        ML_COGNITIVE_MEMORY_DRY_RUN=False,
        ML_COGNITIVE_MEMORY_URL="http://localhost:8080",
    )


@pytest.fixture
def sample_prediction() -> Prediction:
    return Prediction(
        series_id="42",
        predicted_value=25.5,
        confidence_score=0.85,
        trend="up",
        engine_name="taylor",
        horizon_steps=1,
        feature_contributions={"lag_1": 0.6},
        metadata={"explanation": "Upward trend with high confidence"},
        audit_trace_id="trace-abc",
    )


@pytest.fixture
def sample_anomaly() -> AnomalyResult:
    return AnomalyResult(
        series_id="42",
        is_anomaly=True,
        score=0.87,
        method_votes={"isolation_forest": 0.9},
        confidence=0.88,
        explanation="Sudden variance increase",
        severity=AnomalySeverity.HIGH,
        context={"regime": "active"},
        audit_trace_id="trace-def",
    )


def _make_decorator(
    mock_inner: MagicMock,
    mock_cognitive: MagicMock,
    flags: FeatureFlags,
) -> CognitiveStorageDecorator:
    return CognitiveStorageDecorator(mock_inner, mock_cognitive, flags)


# ---------------------------------------------------------------------------
# Test: isinstance check
# ---------------------------------------------------------------------------

class TestDecoratorContract:
    def test_is_storage_port(self, mock_inner, mock_cognitive, flags_enabled):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        assert isinstance(dec, StoragePort)


# ---------------------------------------------------------------------------
# Test: read pass-through (no cognitive involvement)
# ---------------------------------------------------------------------------

class TestReadPassThrough:
    def test_load_sensor_window_delegates(
        self, mock_inner, mock_cognitive, flags_enabled
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        result = dec.load_sensor_window(1, 100)
        mock_inner.load_sensor_window.assert_called_once_with(1, 100)
        assert result == mock_inner.load_sensor_window.return_value

    def test_list_active_sensor_ids_delegates(
        self, mock_inner, mock_cognitive, flags_enabled
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        result = dec.list_active_sensor_ids()
        mock_inner.list_active_sensor_ids.assert_called_once()
        assert result == [1, 2, 3]

    def test_get_latest_prediction_delegates(
        self, mock_inner, mock_cognitive, flags_enabled
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        dec.get_latest_prediction(1)
        mock_inner.get_latest_prediction.assert_called_once_with(1)

    def test_get_sensor_metadata_delegates(
        self, mock_inner, mock_cognitive, flags_enabled
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        dec.get_sensor_metadata(1)
        mock_inner.get_sensor_metadata.assert_called_once_with(1)

    def test_get_device_id_for_sensor_delegates(
        self, mock_inner, mock_cognitive, flags_enabled
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        result = dec.get_device_id_for_sensor(1)
        mock_inner.get_device_id_for_sensor.assert_called_once_with(1)
        assert result == 7

    def test_reads_never_call_cognitive(
        self, mock_inner, mock_cognitive, flags_enabled
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        dec.load_sensor_window(1)
        dec.list_active_sensor_ids()
        dec.get_latest_prediction(1)
        dec.get_sensor_metadata(1)
        dec.get_device_id_for_sensor(1)
        mock_cognitive.assert_not_called()


# ---------------------------------------------------------------------------
# Test: save_prediction dual-write
# ---------------------------------------------------------------------------

class TestSavePrediction:
    def test_calls_inner_first_and_returns_id(
        self, mock_inner, mock_cognitive, flags_enabled, sample_prediction
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        result = dec.save_prediction(sample_prediction)
        assert result == 42
        mock_inner.save_prediction.assert_called_once_with(sample_prediction)

    def test_fires_remember_explanation_after_sql(
        self, mock_inner, mock_cognitive, flags_enabled, sample_prediction
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        dec.save_prediction(sample_prediction)

        mock_cognitive.remember_explanation.assert_called_once()
        call_args = mock_cognitive.remember_explanation.call_args
        assert call_args[0][0] is sample_prediction
        assert call_args[0][1] == 42  # source_record_id from SQL
        assert call_args[1]["domain_name"] == "iot"

    def test_uses_metadata_explanation_text(
        self, mock_inner, mock_cognitive, flags_enabled, sample_prediction
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        dec.save_prediction(sample_prediction)

        call_args = mock_cognitive.remember_explanation.call_args
        assert call_args[1]["explanation_text"] == "Upward trend with high confidence"

    def test_sql_failure_propagates_no_cognitive_call(
        self, mock_inner, mock_cognitive, flags_enabled, sample_prediction
    ):
        mock_inner.save_prediction.side_effect = RuntimeError("SQL error")
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)

        with pytest.raises(RuntimeError, match="SQL error"):
            dec.save_prediction(sample_prediction)

        mock_cognitive.remember_explanation.assert_not_called()


# ---------------------------------------------------------------------------
# Test: save_anomaly_event dual-write
# ---------------------------------------------------------------------------

class TestSaveAnomalyEvent:
    def test_calls_inner_first_and_returns_id(
        self, mock_inner, mock_cognitive, flags_enabled, sample_anomaly
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        result = dec.save_anomaly_event(sample_anomaly, prediction_id=42)
        assert result == 99
        mock_inner.save_anomaly_event.assert_called_once_with(
            sample_anomaly, 42
        )

    def test_fires_remember_anomaly_after_sql(
        self, mock_inner, mock_cognitive, flags_enabled, sample_anomaly
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        dec.save_anomaly_event(sample_anomaly)

        mock_cognitive.remember_anomaly.assert_called_once()
        call_args = mock_cognitive.remember_anomaly.call_args
        assert call_args[0][0] is sample_anomaly
        assert call_args[0][1] == 99  # source_record_id from SQL
        assert call_args[1]["event_code"] == "ANOMALY_DETECTED"
        assert call_args[1]["domain_name"] == "iot"

    def test_serializes_context_to_operational_context(
        self, mock_inner, mock_cognitive, flags_enabled, sample_anomaly
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)
        dec.save_anomaly_event(sample_anomaly)

        call_args = mock_cognitive.remember_anomaly.call_args
        ctx = call_args[1]["operational_context"]
        assert "regime" in ctx
        assert "active" in ctx

    def test_sql_failure_propagates_no_cognitive_call(
        self, mock_inner, mock_cognitive, flags_enabled, sample_anomaly
    ):
        mock_inner.save_anomaly_event.side_effect = RuntimeError("SQL error")
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)

        with pytest.raises(RuntimeError, match="SQL error"):
            dec.save_anomaly_event(sample_anomaly)

        mock_cognitive.remember_anomaly.assert_not_called()


# ---------------------------------------------------------------------------
# Test: cognitive failure is swallowed
# ---------------------------------------------------------------------------

class TestCognitiveFailSafe:
    def test_prediction_cognitive_error_does_not_propagate(
        self, mock_inner, mock_cognitive, flags_enabled, sample_prediction
    ):
        mock_cognitive.remember_explanation.side_effect = Exception("Weaviate down")
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)

        result = dec.save_prediction(sample_prediction)
        assert result == 42  # SQL result still returned

    def test_anomaly_cognitive_error_does_not_propagate(
        self, mock_inner, mock_cognitive, flags_enabled, sample_anomaly
    ):
        mock_cognitive.remember_anomaly.side_effect = Exception("Weaviate down")
        dec = _make_decorator(mock_inner, mock_cognitive, flags_enabled)

        result = dec.save_anomaly_event(sample_anomaly)
        assert result == 99  # SQL result still returned


# ---------------------------------------------------------------------------
# Test: flag disabled → no cognitive calls
# ---------------------------------------------------------------------------

class TestFlagDisabled:
    def test_save_prediction_no_cognitive_call(
        self, mock_inner, mock_cognitive, flags_disabled, sample_prediction
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_disabled)
        result = dec.save_prediction(sample_prediction)
        assert result == 42
        mock_inner.save_prediction.assert_called_once()
        mock_cognitive.remember_explanation.assert_not_called()

    def test_save_anomaly_no_cognitive_call(
        self, mock_inner, mock_cognitive, flags_disabled, sample_anomaly
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_disabled)
        result = dec.save_anomaly_event(sample_anomaly)
        assert result == 99
        mock_inner.save_anomaly_event.assert_called_once()
        mock_cognitive.remember_anomaly.assert_not_called()


# ---------------------------------------------------------------------------
# Test: async mode fires in background thread
# ---------------------------------------------------------------------------

class TestAsyncMode:
    def test_save_prediction_fires_in_thread(
        self, mock_inner, mock_cognitive, flags_async, sample_prediction
    ):
        event = threading.Event()
        original_remember = mock_cognitive.remember_explanation

        def _side_effect(*args, **kwargs):
            original_remember(*args, **kwargs)
            event.set()

        mock_cognitive.remember_explanation = MagicMock(side_effect=_side_effect)

        dec = _make_decorator(mock_inner, mock_cognitive, flags_async)
        result = dec.save_prediction(sample_prediction)

        assert result == 42
        # Wait for background thread
        assert event.wait(timeout=2.0), "Cognitive write did not fire in time"
        mock_cognitive.remember_explanation.assert_called_once()

    def test_save_anomaly_fires_in_thread(
        self, mock_inner, mock_cognitive, flags_async, sample_anomaly
    ):
        event = threading.Event()
        original_remember = mock_cognitive.remember_anomaly

        def _side_effect(*args, **kwargs):
            original_remember(*args, **kwargs)
            event.set()

        mock_cognitive.remember_anomaly = MagicMock(side_effect=_side_effect)

        dec = _make_decorator(mock_inner, mock_cognitive, flags_async)
        result = dec.save_anomaly_event(sample_anomaly)

        assert result == 99
        assert event.wait(timeout=2.0), "Cognitive write did not fire in time"
        mock_cognitive.remember_anomaly.assert_called_once()

    def test_async_cognitive_error_does_not_propagate(
        self, mock_inner, mock_cognitive, flags_async, sample_prediction
    ):
        event = threading.Event()

        def _side_effect(*args, **kwargs):
            event.set()
            raise Exception("Weaviate timeout")

        mock_cognitive.remember_explanation = MagicMock(side_effect=_side_effect)

        dec = _make_decorator(mock_inner, mock_cognitive, flags_async)
        result = dec.save_prediction(sample_prediction)

        assert result == 42
        assert event.wait(timeout=2.0)
        # No exception propagated


# ---------------------------------------------------------------------------
# Test: regression — ML pipeline works identically with flag off
# ---------------------------------------------------------------------------

class TestRegressionFlagOff:
    """Ensures the decorator is a transparent pass-through when disabled."""

    def test_full_pipeline_sequence_with_flag_off(
        self, mock_inner, mock_cognitive, flags_disabled,
        sample_prediction, sample_anomaly,
    ):
        dec = _make_decorator(mock_inner, mock_cognitive, flags_disabled)

        # Simulate a full pipeline cycle
        sensors = dec.list_active_sensor_ids()
        assert sensors == [1, 2, 3]

        window = dec.load_sensor_window(1, 500)
        assert window.sensor_id == 1

        pred_id = dec.save_prediction(sample_prediction)
        assert pred_id == 42

        latest = dec.get_latest_prediction(1)
        assert latest is None

        anom_id = dec.save_anomaly_event(sample_anomaly, prediction_id=pred_id)
        assert anom_id == 99

        meta = dec.get_sensor_metadata(1)
        assert meta == {"sensor_id": 1}

        device = dec.get_device_id_for_sensor(1)
        assert device == 7

        # Cognitive port was NEVER touched
        mock_cognitive.remember_explanation.assert_not_called()
        mock_cognitive.remember_anomaly.assert_not_called()
        mock_cognitive.remember_pattern.assert_not_called()
        mock_cognitive.remember_decision.assert_not_called()

    def test_decorator_returns_exact_same_values_as_inner(
        self, mock_inner, mock_cognitive, flags_disabled, sample_prediction
    ):
        """Verify the decorator doesn't alter return values."""
        dec = _make_decorator(mock_inner, mock_cognitive, flags_disabled)

        mock_inner.save_prediction.return_value = 12345
        assert dec.save_prediction(sample_prediction) == 12345

        mock_inner.save_prediction.return_value = 0
        assert dec.save_prediction(sample_prediction) == 0


# ---------------------------------------------------------------------------
# Test: factory function (requires sqlalchemy)
# ---------------------------------------------------------------------------

try:
    import sqlalchemy as _sa
    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False

_skip_no_sqlalchemy = pytest.mark.skipif(
    not _HAS_SQLALCHEMY,
    reason="sqlalchemy not installed (factory tests require it)",
)


@_skip_no_sqlalchemy
class TestBuildStorageFactory:
    def test_returns_raw_sql_when_disabled(self):
        from iot_machine_learning.infrastructure.adapters.cognitive_storage_factory import (
            build_storage,
        )
        flags = FeatureFlags(ML_ENABLE_COGNITIVE_MEMORY=False)
        mock_conn = MagicMock()

        with patch(
            "iot_machine_learning.infrastructure.adapters.cognitive_storage_factory.SqlServerStorageAdapter"
        ) as MockSql:
            storage = build_storage(mock_conn, flags)
            MockSql.assert_called_once_with(mock_conn)
            assert storage is MockSql.return_value

    def test_returns_decorator_when_enabled_with_url(self):
        from iot_machine_learning.infrastructure.adapters.cognitive_storage_factory import (
            build_storage,
        )
        flags = FeatureFlags(
            ML_ENABLE_COGNITIVE_MEMORY=True,
            ML_COGNITIVE_MEMORY_URL="http://localhost:8080",
            ML_COGNITIVE_MEMORY_DRY_RUN=False,
        )
        mock_conn = MagicMock()

        with patch(
            "iot_machine_learning.infrastructure.adapters.cognitive_storage_factory.SqlServerStorageAdapter"
        ), patch(
            "iot_machine_learning.infrastructure.adapters.cognitive_storage_factory.WeaviateCognitiveAdapter"
        ):
            storage = build_storage(mock_conn, flags)
            assert isinstance(storage, CognitiveStorageDecorator)

    def test_returns_decorator_with_null_cognitive_when_no_url(self):
        from iot_machine_learning.infrastructure.adapters.cognitive_storage_factory import (
            build_storage,
        )
        flags = FeatureFlags(
            ML_ENABLE_COGNITIVE_MEMORY=True,
            ML_COGNITIVE_MEMORY_URL="",
        )
        mock_conn = MagicMock()

        with patch(
            "iot_machine_learning.infrastructure.adapters.cognitive_storage_factory.SqlServerStorageAdapter"
        ):
            storage = build_storage(mock_conn, flags)
            assert isinstance(storage, CognitiveStorageDecorator)
