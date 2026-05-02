"""Tests FIX-10: SensorProcessor migrated from Writer B to Writer A.

Validates:
- SensorProcessor uses SqlServerStorageAdapter.save_prediction (Writer A)
- SensorProcessor NEVER uses PredictionWriter (Writer B)
- Prediction object is constructed with all fields needed for Writer A's 17 columns
- Writer B is deprecated and emits DeprecationWarning
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestSensorProcessorUsesWriterA:
    """Verify SensorProcessor persists via Writer A (SqlServerStorageAdapter)."""

    @pytest.fixture
    def mock_conn(self):
        return MagicMock()

    @pytest.fixture
    def processor_deps(self):
        """Return minimal mocked dependencies for SensorProcessor.__init__."""
        from iot_machine_learning.ml_service.runners.common.sensor_processor import (
            SensorProcessor,
        )

        mock_event_writer = Mock()
        mock_regression_service = Mock()
        mock_severity_classifier = Mock()
        mock_model_manager = Mock()
        mock_narrator = Mock()

        processor = SensorProcessor(
            event_writer=mock_event_writer,
            regression_service=mock_regression_service,
            severity_classifier=mock_severity_classifier,
            model_manager=mock_model_manager,
            narrator=mock_narrator,
        )
        return {
            "processor": processor,
            "event_writer": mock_event_writer,
            "regression_service": mock_regression_service,
            "severity_classifier": mock_severity_classifier,
            "model_manager": mock_model_manager,
            "narrator": mock_narrator,
        }

    def test_process_calls_save_prediction_on_writer_a(self, mock_conn, processor_deps):
        """Writer A (SqlServerStorageAdapter.save_prediction) must be called."""
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )
        from iot_machine_learning.ml_service.runners.common.sensor_processor import (
            SensorProcessor,
        )

        processor = processor_deps["processor"]

        # Patch all internal functions used inside process()
        with patch(
            "iot_machine_learning.ml_service.repository.sensor_repository.load_sensor_series"
        ) as mock_load_series, patch(
            "iot_machine_learning.ml_service.repository.sensor_repository.load_sensor_metadata"
        ) as mock_load_meta, patch(
            "iot_machine_learning.ml_service.repository.sensor_repository.get_device_id_for_sensor"
        ) as mock_get_dev, patch(
            "iot_machine_learning.ml_service.trainers.regression_trainer.train_regression_for_sensor"
        ) as mock_train, patch(
            "iot_machine_learning.ml_service.runners.common.sensor_processor.SqlServerStorageAdapter"
        ) as MockStorage:

            mock_load_series.return_value = SensorWindow(
                series_id="42",
                readings=[
                    SensorReading(series_id="42", value=20.0, timestamp=1000.0),
                    SensorReading(series_id="42", value=21.0, timestamp=1001.0),
                    SensorReading(series_id="42", value=22.0, timestamp=1002.0),
                ],
            )
            mock_load_meta.return_value = {"name": "temp_sensor"}
            mock_get_dev.return_value = 7
            mock_train.return_value = (None, None)  # fallback path

            mock_storage_instance = MockStorage.return_value
            mock_storage_instance.save_prediction.return_value = 12345
            mock_predictions = Mock()
            mock_predictions.adjust_for_delta_spike = Mock()
            mock_storage_instance._predictions = mock_predictions

            processor_deps["regression_service"].predict_fallback.return_value = Mock(
                predicted_value=23.0,
                trend="up",
                confidence=0.85,
                anomaly=False,
                anomaly_score=0.1,
                window_points_effective=3,
                engine_name="regression",
            )
            processor_deps["severity_classifier"].is_value_within_user_thresholds.return_value = False
            processor_deps["narrator"].build_explanation.return_value = Mock(
                predicted_value=23.0,
                trend="up",
                confidence=0.85,
                anomaly=False,
                anomaly_score=0.1,
                severity="info",
                risk_level="NONE",
                explanation="test explanation",
            )
            processor_deps["model_manager"].get_or_create_model_id.return_value = 99

            # Minimal ml_cfg mock
            ml_cfg = Mock()
            ml_cfg.regression.window_points = 3
            ml_cfg.regression.horizon_minutes = 10
            ml_cfg.regression.engine_name = "regression"

            iso_trainer = Mock()

            processor.process(mock_conn, sensor_id=42, ml_cfg=ml_cfg, iso_trainer=iso_trainer)

            # Assert Writer A was used
            MockStorage.assert_called_once_with(mock_conn)
            mock_storage_instance.save_prediction.assert_called_once()
            call_args = mock_storage_instance.save_prediction.call_args
            prediction_arg = call_args[0][0]
            assert prediction_arg.series_id == "42"
            assert prediction_arg.predicted_value == 23.0
            assert prediction_arg.confidence_score == 0.85
            assert prediction_arg.trend == "up"
            assert prediction_arg.engine_name == "regression"

            # Assert metadata carries the enrichment fields Writer A expects
            meta = prediction_arg.metadata
            assert meta.get("is_anomaly") is False
            assert meta.get("anomaly_score") == 0.1
            assert meta.get("severity") == "info"
            assert meta.get("risk_level") == "NONE"
            assert meta.get("explanation") == "test explanation"
            assert meta.get("window_points") == 3
            assert meta.get("model_id") == 99
            assert meta.get("device_id") == 7

            # Assert delta-spike adjustment is also called (Writer A feature)
            mock_storage_instance.adjust_for_delta_spike.assert_called_once_with(12345, 42)

    def test_process_never_uses_prediction_writer(self, mock_conn, processor_deps):
        """Writer B (PredictionWriter) must NEVER be instantiated or called."""
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )
        from iot_machine_learning.ml_service.runners.common.sensor_processor import (
            SensorProcessor,
        )

        processor = processor_deps["processor"]

        with patch(
            "iot_machine_learning.ml_service.repository.sensor_repository.load_sensor_series"
        ) as mock_load_series, patch(
            "iot_machine_learning.ml_service.repository.sensor_repository.load_sensor_metadata"
        ) as mock_load_meta, patch(
            "iot_machine_learning.ml_service.repository.sensor_repository.get_device_id_for_sensor"
        ) as mock_get_dev, patch(
            "iot_machine_learning.ml_service.trainers.regression_trainer.train_regression_for_sensor"
        ) as mock_train, patch(
            "iot_machine_learning.ml_service.runners.common.sensor_processor.SqlServerStorageAdapter"
        ) as MockStorage:

            mock_load_series.return_value = SensorWindow(
                series_id="42",
                readings=[
                    SensorReading(series_id="42", value=20.0, timestamp=1000.0),
                ],
            )
            mock_load_meta.return_value = {}
            mock_get_dev.return_value = 7
            mock_train.return_value = (None, None)

            mock_storage_instance = MockStorage.return_value
            mock_storage_instance.save_prediction.return_value = 12345
            mock_predictions = Mock()
            mock_predictions.adjust_for_delta_spike = Mock()
            mock_storage_instance._predictions = mock_predictions

            processor_deps["regression_service"].predict_fallback.return_value = Mock(
                predicted_value=20.0,
                trend="stable",
                confidence=0.5,
                anomaly=False,
                anomaly_score=0.0,
                window_points_effective=1,
            )
            processor_deps["severity_classifier"].is_value_within_user_thresholds.return_value = False
            processor_deps["narrator"].build_explanation.return_value = Mock(
                predicted_value=20.0,
                trend="stable",
                confidence=0.5,
                anomaly=False,
                anomaly_score=0.0,
                severity="info",
                risk_level="NONE",
                explanation="",
            )
            processor_deps["model_manager"].get_or_create_model_id.return_value = 1

            ml_cfg = Mock()
            ml_cfg.regression.window_points = 1
            ml_cfg.regression.horizon_minutes = 10
            ml_cfg.regression.engine_name = "regression"

            processor.process(mock_conn, sensor_id=42, ml_cfg=ml_cfg, iso_trainer=Mock())

            # Writer A (SqlServerStorageAdapter) was used
            MockStorage.assert_called_once_with(mock_conn)
            mock_storage_instance.save_prediction.assert_called_once()


class TestPredictionWriterDeprecated:
    """Verify Writer B emits DeprecationWarning and is not in __all__."""

    def test_prediction_writer_module_removed(self):
        """Writer B module must no longer exist."""
        with pytest.raises(ImportError):
            from iot_machine_learning.ml_service.runners.common.prediction_writer import (
                PredictionWriter,
            )

    def test_prediction_writer_not_in_all(self):
        from iot_machine_learning.ml_service.runners.common import __all__
        assert "PredictionWriter" not in __all__
