"""Regression tests for performance fixes H-ML-4, H-ML-8, H-ING-2.

H-ML-4: Enterprise preloaded data path (no double SQL query)
H-ML-8: Stream consumer uses sliding window data (no SQL reload)
H-ING-2: Async processor decouples paho thread from SP
"""

from __future__ import annotations

import queue
import time
import threading
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

try:
    from iot_ingest_services.ingest_api.mqtt.async_processor import (
        AsyncReadingProcessor,
        create_async_processor,
    )
    _ASYNC_PROC_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _ASYNC_PROC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(sensor_id: int = 1, n: int = 5) -> SensorWindow:
    readings = [
        SensorReading(sensor_id=sensor_id, value=20.0 + i, timestamp=1000.0 + i)
        for i in range(n)
    ]
    return SensorWindow(sensor_id=sensor_id, readings=readings)


def _make_dto(**overrides):
    defaults = dict(
        series_id="1", predicted_value=21.5, confidence_score=0.85,
        confidence_level="high", trend="rising", engine_name="taylor",
        confidence_interval=None, feature_contributions=None,
        audit_trace_id="abc123", memory_context=None,
    )
    defaults.update(overrides)

    class FakeDTO:
        pass

    dto = FakeDTO()
    for k, v in defaults.items():
        setattr(dto, k, v)
    return dto


# ===================================================================
# H-ML-4: execute_with_window skips storage.load_sensor_window
# ===================================================================

class TestExecuteWithWindow:
    """PredictSensorValueUseCase.execute_with_window must NOT call storage."""

    def test_execute_with_window_skips_load(self):
        from iot_machine_learning.application.use_cases.predict_sensor_value import (
            PredictSensorValueUseCase,
        )

        mock_storage = MagicMock()
        mock_pred_service = MagicMock()

        # Make predict return a proper Prediction-like object
        fake_prediction = MagicMock()
        fake_prediction.series_id = "1"
        fake_prediction.predicted_value = 21.5
        fake_prediction.confidence_score = 0.85
        fake_prediction.confidence_level = MagicMock(value="high")
        fake_prediction.trend = "rising"
        fake_prediction.engine_name = "taylor"
        fake_prediction.confidence_interval = None
        fake_prediction.feature_contributions = None
        fake_prediction.audit_trace_id = "abc123"
        mock_pred_service.predict.return_value = fake_prediction

        uc = PredictSensorValueUseCase(
            prediction_service=mock_pred_service,
            storage=mock_storage,
        )

        window = _make_window(sensor_id=1, n=5)
        dto = uc.execute_with_window(sensor_window=window)

        # Key assertion: storage.load_sensor_window was NOT called
        mock_storage.load_sensor_window.assert_not_called()

        # prediction_service.predict WAS called with the window
        mock_pred_service.predict.assert_called_once_with(window)

        # DTO has correct values
        assert dto.predicted_value == 21.5
        assert dto.engine_name == "taylor"

    def test_execute_with_window_empty_raises(self):
        from iot_machine_learning.application.use_cases.predict_sensor_value import (
            PredictSensorValueUseCase,
        )
        uc = PredictSensorValueUseCase(
            prediction_service=MagicMock(), storage=MagicMock(),
        )
        empty = SensorWindow(sensor_id=1, readings=[])
        with pytest.raises(ValueError, match="no tiene lecturas"):
            uc.execute_with_window(sensor_window=empty)

    def test_execute_with_window_persists(self):
        from iot_machine_learning.application.use_cases.predict_sensor_value import (
            PredictSensorValueUseCase,
        )
        mock_storage = MagicMock()
        mock_pred_service = MagicMock()
        fake_prediction = MagicMock()
        fake_prediction.series_id = "1"
        fake_prediction.predicted_value = 21.5
        fake_prediction.confidence_score = 0.85
        fake_prediction.confidence_level = MagicMock(value="high")
        fake_prediction.trend = "rising"
        fake_prediction.engine_name = "baseline"
        fake_prediction.confidence_interval = None
        fake_prediction.feature_contributions = None
        fake_prediction.audit_trace_id = "t1"
        mock_pred_service.predict.return_value = fake_prediction

        uc = PredictSensorValueUseCase(
            prediction_service=mock_pred_service, storage=mock_storage,
        )
        uc.execute_with_window(sensor_window=_make_window())

        mock_storage.save_prediction.assert_called_once_with(fake_prediction)


# ===================================================================
# H-ML-4: EnterprisePredictionAdapter.predict_with_window
# ===================================================================

class TestAdapterPredictWithWindow:
    def test_predict_with_window_calls_execute_with_window(self):
        from iot_machine_learning.ml_service.runners.adapters.enterprise_prediction import (
            EnterprisePredictionAdapter,
        )
        mock_uc = MagicMock()
        mock_uc.execute_with_window.return_value = _make_dto()
        mock_audit = MagicMock()
        mock_storage = MagicMock()

        adapter = EnterprisePredictionAdapter(
            storage=mock_storage, use_case=mock_uc, audit=mock_audit,
        )
        window = _make_window(sensor_id=42, n=10)
        result = adapter.predict_with_window(sensor_window=window)

        mock_uc.execute_with_window.assert_called_once_with(sensor_window=window)
        mock_uc.execute.assert_not_called()
        assert result.predicted_value == 21.5

    def test_predict_with_window_fallback_on_error(self):
        from iot_machine_learning.ml_service.runners.adapters.enterprise_prediction import (
            EnterprisePredictionAdapter,
        )
        mock_uc = MagicMock()
        mock_uc.execute_with_window.side_effect = RuntimeError("boom")
        mock_audit = MagicMock()
        mock_storage = MagicMock()

        with patch(
            "iot_machine_learning.ml_service.runners.adapters.fallback_baseline.fallback_to_baseline"
        ) as mock_fallback:
            mock_fallback.return_value = MagicMock(
                predicted_value=20.0, confidence=0.5,
            )
            adapter = EnterprisePredictionAdapter(
                storage=mock_storage, use_case=mock_uc, audit=mock_audit,
            )
            result = adapter.predict_with_window(sensor_window=_make_window())

            mock_fallback.assert_called_once()
            assert result.predicted_value == 20.0


# ===================================================================
# H-ML-8: Stream consumer builds SensorWindow from sliding window
# ===================================================================

class TestStreamConsumerSlidingWindow:
    def test_build_sensor_window_from_store(self):
        from iot_machine_learning.ml_service.consumers.stream_consumer import (
            ReadingsStreamConsumer,
        )
        from iot_machine_learning.ml_service.consumers.sliding_window import Reading

        consumer = ReadingsStreamConsumer(min_window=3, max_window=10)

        for i in range(5):
            consumer._store.append(
                Reading(sensor_id=7, value=20.0 + i, timestamp=1000.0 + i, timestamp_iso="")
            )

        window = consumer._build_sensor_window(7)
        assert window is not None
        assert window.sensor_id == 7
        assert window.size == 5
        assert window.values == [20.0, 21.0, 22.0, 23.0, 24.0]

    def test_build_sensor_window_returns_none_below_min(self):
        from iot_machine_learning.ml_service.consumers.stream_consumer import (
            ReadingsStreamConsumer,
        )
        from iot_machine_learning.ml_service.consumers.sliding_window import Reading

        consumer = ReadingsStreamConsumer(min_window=5, max_window=10)
        consumer._store.append(
            Reading(sensor_id=7, value=20.0, timestamp=1000.0, timestamp_iso="")
        )

        window = consumer._build_sensor_window(7)
        assert window is None

    def test_build_sensor_window_filters_nan(self):
        from iot_machine_learning.ml_service.consumers.stream_consumer import (
            ReadingsStreamConsumer,
        )
        from iot_machine_learning.ml_service.consumers.sliding_window import Reading

        consumer = ReadingsStreamConsumer(min_window=2, max_window=10)
        consumer._store.append(
            Reading(sensor_id=7, value=20.0, timestamp=1000.0, timestamp_iso="")
        )
        consumer._store.append(
            Reading(sensor_id=7, value=21.0, timestamp=1001.0, timestamp_iso="")
        )
        # NaN value — should be filtered
        consumer._store.append(
            Reading(sensor_id=7, value=float("nan"), timestamp=1002.0, timestamp_iso="")
        )

        window = consumer._build_sensor_window(7)
        assert window is not None
        assert window.size == 2  # NaN filtered out


# ===================================================================
# H-ING-2: AsyncReadingProcessor
# ===================================================================

@pytest.mark.skipif(not _ASYNC_PROC_AVAILABLE, reason="iot_ingest_services not importable")
class TestAsyncReadingProcessor:
    def test_enqueue_returns_immediately(self):
        mock_processor = MagicMock()
        mock_processor.process.side_effect = lambda p: time.sleep(0.05)

        ap = AsyncReadingProcessor(
            processor=mock_processor, max_queue_size=100, num_workers=2,
        )
        ap.start()

        try:
            payload = MagicMock()
            payload.sensor_id_int = 1

            t0 = time.monotonic()
            result = ap.enqueue(payload)
            elapsed = time.monotonic() - t0

            assert result is True
            assert elapsed < 0.01

            time.sleep(0.2)
            mock_processor.process.assert_called_once_with(payload)
        finally:
            ap.stop(drain=False)

    def test_enqueue_drops_when_full(self):
        mock_processor = MagicMock()
        mock_processor.process.side_effect = lambda p: time.sleep(10)

        ap = AsyncReadingProcessor(
            processor=mock_processor, max_queue_size=2, num_workers=1,
        )
        ap.start()

        try:
            for _ in range(5):
                ap.enqueue(MagicMock())
            time.sleep(0.1)

            metrics = ap.metrics
            assert metrics["dropped"] > 0
        finally:
            ap.stop(drain=False)

    def test_metrics_tracking(self):
        mock_processor = MagicMock()
        ap = AsyncReadingProcessor(
            processor=mock_processor, max_queue_size=100, num_workers=2,
        )
        ap.start()

        try:
            for _ in range(5):
                ap.enqueue(MagicMock())

            time.sleep(0.5)
            m = ap.metrics
            assert m["enqueued"] == 5
            assert m["processed"] == 5
            assert m["errors"] == 0
        finally:
            ap.stop(drain=False)

    def test_create_async_processor_disabled(self):
        with patch.dict("os.environ", {"ML_MQTT_ASYNC_PROCESSING": "false"}):
            result = create_async_processor(MagicMock())
            assert result is None


# ===================================================================
# Feature flags defaults
# ===================================================================

class TestPerformanceFeatureFlags:
    def test_defaults_all_enabled(self):
        ff = FeatureFlags()
        assert ff.ML_ENTERPRISE_USE_PRELOADED_DATA is True
        assert ff.ML_STREAM_USE_SLIDING_WINDOW is True
        assert ff.ML_MQTT_ASYNC_PROCESSING is True
        assert ff.ML_MQTT_QUEUE_SIZE == 1000
        assert ff.ML_MQTT_NUM_WORKERS == 4

    def test_can_disable_preloaded(self):
        ff = FeatureFlags(ML_ENTERPRISE_USE_PRELOADED_DATA=False)
        assert ff.ML_ENTERPRISE_USE_PRELOADED_DATA is False

    def test_can_disable_async(self):
        ff = FeatureFlags(ML_MQTT_ASYNC_PROCESSING=False)
        assert ff.ML_MQTT_ASYNC_PROCESSING is False
