"""INF-5 — Guard contra dual prediction paths.

Valida que ML_STREAM_PREDICTIONS_ENABLED y batch runner activos
simultáneamente producen RuntimeError en startup (fail-fast).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from iot_machine_learning.ml_service.config.loader import get_feature_flags, reset_feature_flags
from iot_machine_learning.ml_service.main import _validate_prediction_paths


class TestPredictionPathGuard:
    """Valida la exclusión mutua entre stream predictions y batch runner."""

    def test_stream_only_allowed(self):
        """Stream activo + batch inactivo = OK."""
        env = {
            "ML_STREAM_PREDICTIONS_ENABLED": "true",
            "ZENIN_QUEUE_POLLER_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            reset_feature_flags()
            _validate_prediction_paths()  # no raise

    def test_batch_only_allowed(self):
        """Batch activo + stream inactivo = OK."""
        env = {
            "ML_STREAM_PREDICTIONS_ENABLED": "false",
            "ZENIN_QUEUE_POLLER_ENABLED": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            reset_feature_flags()
            _validate_prediction_paths()  # no raise

    def test_both_active_raises(self):
        """Ambos activos → RuntimeError con mensaje INF-5."""
        env = {
            "ML_STREAM_PREDICTIONS_ENABLED": "true",
            "ZENIN_QUEUE_POLLER_ENABLED": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            reset_feature_flags()
            with pytest.raises(RuntimeError, match="INF-5"):
                _validate_prediction_paths()

    def test_stream_with_batch_sensors_raises(self):
        """Stream activo + ML_BATCH_ENTERPRISE_SENSORS configurado → RuntimeError."""
        env = {
            "ML_STREAM_PREDICTIONS_ENABLED": "true",
            "ZENIN_QUEUE_POLLER_ENABLED": "false",
            "ML_BATCH_ENTERPRISE_SENSORS": "1,2,3",
        }
        with patch.dict(os.environ, env, clear=False):
            reset_feature_flags()
            with pytest.raises(RuntimeError, match="INF-5"):
                _validate_prediction_paths()
