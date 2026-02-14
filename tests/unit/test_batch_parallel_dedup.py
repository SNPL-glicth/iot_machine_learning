"""Tests for E-4 (parallel batch runner) and E-2 (prediction dedup).

Covers:
- Parallel processing with ThreadPoolExecutor
- Sensor fault isolation (1 failure doesn't break batch)
- Feature flag ML_BATCH_PARALLEL_WORKERS
- Feature flag ML_STREAM_PREDICTIONS_ENABLED
- Stream consumer skips predictions when flag is False
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

# ======================================================================
# E-4: Parallel Batch Runner
# ======================================================================

try:
    from iot_ingest_services.jobs.batch.runner import (
        _process_sensor,
        run_once,
    )
    from iot_ingest_services.jobs.batch.config import RunnerConfig
    _BATCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _BATCH_AVAILABLE = False


def _make_cfg():
    return RunnerConfig(window=20, horizon_minutes=5, dedupe_minutes=30,
                        sleep_seconds=60, once=True)


@pytest.mark.skipif(not _BATCH_AVAILABLE, reason="iot_ingest_services not importable")
class TestParallelBatchRunner:

    @patch("iot_ingest_services.jobs.batch.runner.get_engine")
    @patch("iot_ingest_services.jobs.batch.runner.list_active_sensors")
    @patch("iot_ingest_services.jobs.batch.runner._process_sensor")
    def test_all_sensors_processed(self, mock_process, mock_list, mock_engine):
        """All sensors should be submitted to the pool."""
        mock_list.return_value = [1, 2, 3, 4, 5]
        mock_process.return_value = "baseline"
        mock_engine.return_value = MagicMock()

        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        flags = FeatureFlags()

        with patch.dict("os.environ", {"ML_BATCH_PARALLEL_WORKERS": "2"}):
            run_once(_make_cfg(), flags=flags)

        assert mock_process.call_count == 5

    @patch("iot_ingest_services.jobs.batch.runner.get_engine")
    @patch("iot_ingest_services.jobs.batch.runner.list_active_sensors")
    @patch("iot_ingest_services.jobs.batch.runner._process_sensor")
    def test_sensor_isolation(self, mock_process, mock_list, mock_engine):
        """Failure in one sensor should NOT break the batch."""
        mock_list.return_value = [1, 2, 3]
        mock_process.side_effect = [
            "baseline",
            Exception("SQL error on sensor 2"),
            "baseline",
        ]
        mock_engine.return_value = MagicMock()

        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        # Should NOT raise
        run_once(_make_cfg(), flags=FeatureFlags())

        assert mock_process.call_count == 3

    @patch("iot_ingest_services.jobs.batch.runner.get_engine")
    @patch("iot_ingest_services.jobs.batch.runner.list_active_sensors")
    @patch("iot_ingest_services.jobs.batch.runner._process_sensor")
    def test_empty_sensor_list(self, mock_process, mock_list, mock_engine):
        """Empty sensor list should complete without error."""
        mock_list.return_value = []
        mock_engine.return_value = MagicMock()

        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        run_once(_make_cfg(), flags=FeatureFlags())

        mock_process.assert_not_called()

    @patch("iot_ingest_services.jobs.batch.runner.get_engine")
    @patch("iot_ingest_services.jobs.batch.runner.list_active_sensors")
    @patch("iot_ingest_services.jobs.batch.runner._process_sensor")
    def test_default_workers_is_sequential(self, mock_process, mock_list, mock_engine):
        """Default ML_BATCH_PARALLEL_WORKERS=1 means sequential."""
        mock_list.return_value = [1, 2]
        mock_process.return_value = "baseline"
        mock_engine.return_value = MagicMock()

        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        # No env var set → default 1 worker
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("ML_BATCH_PARALLEL_WORKERS", None)
            run_once(_make_cfg(), flags=FeatureFlags())

        assert mock_process.call_count == 2


@pytest.mark.skipif(not _BATCH_AVAILABLE, reason="iot_ingest_services not importable")
class TestProcessSensorIsolation:

    @patch("iot_ingest_services.jobs.batch.runner.ensure_watermark")
    @patch("iot_ingest_services.jobs.batch.runner.get_last_reading_id", return_value=None)
    @patch("iot_ingest_services.jobs.batch.runner.get_sensor_max_reading_id", return_value=100)
    @patch("iot_ingest_services.jobs.batch.runner.load_recent_values", return_value=[1.0, 2.0, 3.0])
    @patch("iot_ingest_services.jobs.batch.runner.get_or_create_active_model_id", return_value=1)
    @patch("iot_ingest_services.jobs.batch.runner.get_device_id_for_sensor", return_value=10)
    @patch("iot_ingest_services.jobs.batch.runner.insert_prediction", return_value=999)
    @patch("iot_ingest_services.jobs.batch.runner.eval_pred_threshold_and_create_event")
    @patch("iot_ingest_services.jobs.batch.runner.update_watermark")
    def test_process_sensor_baseline(self, mock_wm, mock_eval, mock_insert,
                                      mock_dev, mock_model, mock_vals,
                                      mock_max, mock_last, mock_ensure):
        """_process_sensor should complete a full baseline cycle."""
        engine = MagicMock()
        conn = MagicMock()
        engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
        engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        tag = _process_sensor(engine, _make_cfg(), FeatureFlags(), None, 42)

        assert tag == "baseline"
        mock_insert.assert_called_once()
        mock_wm.assert_called_once()

    @patch("iot_ingest_services.jobs.batch.runner.ensure_watermark")
    @patch("iot_ingest_services.jobs.batch.runner.get_last_reading_id", return_value=None)
    @patch("iot_ingest_services.jobs.batch.runner.get_sensor_max_reading_id", return_value=None)
    def test_process_sensor_no_readings(self, mock_max, mock_last, mock_ensure):
        """Sensor with no readings should return None."""
        engine = MagicMock()
        conn = MagicMock()
        engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
        engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        tag = _process_sensor(engine, _make_cfg(), FeatureFlags(), None, 42)

        assert tag is None


# ======================================================================
# E-2: Prediction Dedup — Feature Flag
# ======================================================================

class TestStreamPredictionDedup:

    def test_flag_default_is_false(self):
        """ML_STREAM_PREDICTIONS_ENABLED should default to False (batch only)."""
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert flags.ML_STREAM_PREDICTIONS_ENABLED is False

    def test_flag_can_be_enabled(self):
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        flags = FeatureFlags(ML_STREAM_PREDICTIONS_ENABLED=True)
        assert flags.ML_STREAM_PREDICTIONS_ENABLED is True

    @patch("iot_machine_learning.ml_service.config.feature_flags.get_feature_flags")
    def test_stream_consumer_skips_when_disabled(self, mock_flags):
        """Stream consumer _predict should return early when flag is False."""
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        mock_flags.return_value = FeatureFlags(ML_STREAM_PREDICTIONS_ENABLED=False)

        from iot_machine_learning.ml_service.consumers.stream_consumer import (
            ReadingsStreamConsumer,
        )
        consumer = ReadingsStreamConsumer.__new__(ReadingsStreamConsumer)
        consumer._use_case = MagicMock()
        consumer._store = MagicMock()
        consumer._min_window = 5

        # _predict should return without calling adapter
        consumer._predict(42)

        # The adapter should NOT have been called
        consumer._use_case.get_prediction_adapter.assert_not_called()

    @patch("iot_machine_learning.ml_service.config.feature_flags.get_feature_flags")
    def test_stream_consumer_predicts_when_enabled(self, mock_flags):
        """Stream consumer _predict should proceed when flag is True."""
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        mock_flags.return_value = FeatureFlags(ML_STREAM_PREDICTIONS_ENABLED=True)

        from iot_machine_learning.ml_service.consumers.stream_consumer import (
            ReadingsStreamConsumer,
        )
        consumer = ReadingsStreamConsumer.__new__(ReadingsStreamConsumer)
        consumer._use_case = MagicMock()
        consumer._store = MagicMock()
        consumer._min_window = 5

        # Mock _build_sensor_window to return None (no window)
        consumer._build_sensor_window = MagicMock(return_value=None)

        consumer._predict(42)

        # Should have attempted to get adapter (flag allowed it)
        consumer._use_case.get_prediction_adapter.assert_called_once()


# ======================================================================
# Feature Flag Existence
# ======================================================================

class TestPhase2FeatureFlags:

    def test_batch_parallel_workers_flag(self):
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert flags.ML_BATCH_PARALLEL_WORKERS == 1

    def test_stream_predictions_flag(self):
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert flags.ML_STREAM_PREDICTIONS_ENABLED is False
