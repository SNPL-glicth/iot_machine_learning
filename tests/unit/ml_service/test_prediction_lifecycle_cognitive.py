"""Tests for prediction_lifecycle cognitive orchestrator injection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestPredictionLifecycleCognitive:
    """Verify prediction_lifecycle wires orchestrator with graceful fallback."""

    def test_lifecycle_injects_orchestrator_when_flag_enabled(self):
        """When ML_USE_COGNITIVE_ORCHESTRATOR=True, orchestrator is passed to PredictionService."""
        mock_orchestrator = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.orchestrator = mock_orchestrator

        with patch("iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection.ZeninDbConnection.get_engine") as mock_get_engine, \
             patch("iot_machine_learning.ml_service.config.feature_flags.get_feature_flags") as mock_get_flags, \
             patch("iot_machine_learning.ml_service.runners.wiring.container.BatchEnterpriseContainer") as mock_container_cls, \
             patch("iot_machine_learning.ml_service.api.services.prediction_service.PredictionService") as mock_service_cls:
            # Arrange
            mock_get_engine.return_value = MagicMock()
            mock_flags = MagicMock()
            mock_flags.ML_USE_COGNITIVE_ORCHESTRATOR = True
            mock_get_flags.return_value = mock_flags

            mock_container = MagicMock()
            mock_container.get_prediction_adapter.return_value = MagicMock()
            mock_container.get_cognitive_adapter.return_value = mock_adapter
            mock_container_cls.return_value = mock_container

            mock_service = MagicMock()
            mock_service.predict.return_value = {"predicted_value": 42.0}
            mock_service_cls.return_value = mock_service

            # Import and build factory (full package path for relative imports)
            from iot_machine_learning.ml_service.consumers.prediction_lifecycle import (
                _make_worker_factory,
            )

            factory = _make_worker_factory(MagicMock())
            predict_fn = factory()

            # Create a mock task
            task = MagicMock()
            task.mode = "http"
            task.sensor_id = 7
            task.horizon_minutes = 10
            task.window = 60
            task.dedupe_minutes = 10

            # Act
            result = predict_fn(task)

            # Assert
            call_kwargs = mock_service_cls.call_args.kwargs
            assert call_kwargs.get("cognitive_orchestrator") is mock_orchestrator
            assert result == {"predicted_value": 42.0}

    def test_lifecycle_fallback_when_container_fails(self):
        """If get_cognitive_adapter fails, PredictionService still works with orchestrator=None."""
        with patch("iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection.ZeninDbConnection.get_engine") as mock_get_engine, \
             patch("iot_machine_learning.ml_service.config.feature_flags.get_feature_flags") as mock_get_flags, \
             patch("iot_machine_learning.ml_service.runners.wiring.container.BatchEnterpriseContainer") as mock_container_cls, \
             patch("iot_machine_learning.ml_service.api.services.prediction_service.PredictionService") as mock_service_cls:
            # Arrange
            mock_get_engine.return_value = MagicMock()
            mock_flags = MagicMock()
            mock_flags.ML_USE_COGNITIVE_ORCHESTRATOR = True
            mock_get_flags.return_value = mock_flags

            mock_container = MagicMock()
            mock_container.get_prediction_adapter.return_value = MagicMock()
            mock_container.get_cognitive_adapter.side_effect = RuntimeError("Redis down")
            mock_container_cls.return_value = mock_container

            mock_service = MagicMock()
            mock_service.predict.return_value = {"predicted_value": 21.0}
            mock_service_cls.return_value = mock_service

            from iot_machine_learning.ml_service.consumers.prediction_lifecycle import (
                _make_worker_factory,
            )

            factory = _make_worker_factory(MagicMock())
            predict_fn = factory()

            task = MagicMock()
            task.mode = "http"
            task.sensor_id = 7
            task.horizon_minutes = 10
            task.window = 60
            task.dedupe_minutes = 10

            # Act
            result = predict_fn(task)

            # Assert
            call_kwargs = mock_service_cls.call_args.kwargs
            assert call_kwargs.get("cognitive_orchestrator") is None
            assert result == {"predicted_value": 21.0}
