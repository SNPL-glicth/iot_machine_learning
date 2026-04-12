"""Tests para PredictSensorValueUseCase con ExperimentTracker.

Verifica que la inyección del tracker funciona correctamente
y que el tracking ocurre en cada predicción.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.application.dto.prediction_dto import PredictionDTO
from iot_machine_learning.application.use_cases.predict_sensor_value import (
    PredictSensorValueUseCase,
)
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.domain.ports.experiment_tracker_port import (
    ExperimentTrackerPort,
    NullExperimentTracker,
)
from iot_machine_learning.domain.services.prediction_domain_service import (
    PredictionDomainService,
)


class MockPredictionPort:
    """Mock de PredictionPort para tests."""

    def __init__(self, name: str = "mock_engine"):
        self.name = name

    def can_handle(self, n_points: int) -> bool:
        return n_points >= 1

    def predict(self, window: SensorWindow) -> Prediction:
        return Prediction(
            series_id=str(window.sensor_id),
            predicted_value=42.0,
            confidence_score=0.85,
            trend="stable",
            engine_name=self.name,
            confidence_interval=(40.0, 44.0),
        )


class MockStoragePort:
    """Mock de StoragePort para tests."""

    def load_sensor_window(self, sensor_id: int, limit: int) -> SensorWindow:
        from iot_machine_learning.domain.entities.iot.sensor_reading import Reading, SensorWindow
        readings = [
            Reading(series_id=str(sensor_id), value=float(i), timestamp=1000 + i)
            for i in range(limit)
        ]
        return SensorWindow(series_id=str(sensor_id), readings=readings)

    def save_prediction(self, prediction: Prediction) -> None:
        pass


class MockExperimentTracker(ExperimentTrackerPort):
    """Mock de tracker que registra todas las llamadas."""

    def __init__(self):
        self.metrics_logged: list = []
        self.params_logged: list = []
        self.tags_set: list = []
        self.runs_started: list = []
        self.runs_ended: list = []

    def start_run(self, run_name: Optional[str] = None, tags: Optional[dict] = None):
        self.runs_started.append((run_name, tags))
        return f"mock_run_{len(self.runs_started)}"

    def end_run(self, status: Optional[str] = None):
        self.runs_ended.append(status)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        self.metrics_logged.append((key, value, step))

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        for key, value in metrics.items():
            self.metrics_logged.append((key, value, step))

    def log_param(self, key: str, value):
        self.params_logged.append((key, value))

    def log_params(self, params: dict):
        for key, value in params.items():
            self.params_logged.append((key, value))

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass

    def set_tags(self, tags: dict):
        for key, value in tags.items():
            self.tags_set.append((key, value))

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status)


@pytest.fixture
def mock_use_case_with_tracker():
    """Fixture con use case y tracker mock."""
    storage = MockStoragePort()
    engine = MockPredictionPort("test_engine")
    domain_service = PredictionDomainService(engines=[engine])
    tracker = MockExperimentTracker()

    use_case = PredictSensorValueUseCase(
        prediction_service=domain_service,
        storage=storage,
        experiment_tracker=tracker,
    )

    return use_case, tracker


class TestPredictUseCaseWithTracker:
    """Tests de integración use case + tracker."""

    def test_tracker_injected_and_used(self, mock_use_case_with_tracker):
        """El tracker inyectado recibe las llamadas de tracking."""
        use_case, tracker = mock_use_case_with_tracker

        # Ejecutar predicción
        dto = use_case.execute(sensor_id=1, window_size=10)

        # Verificar que se loguearon métricas
        assert len(tracker.metrics_logged) > 0

        # Verificar que confidence_score está en las métricas
        metric_keys = [m[0] for m in tracker.metrics_logged]
        assert "confidence_score" in metric_keys
        assert "elapsed_ms" in metric_keys

    def test_params_logged(self, mock_use_case_with_tracker):
        """Los parámetros (engine, series_id, window_size) se loguean."""
        use_case, tracker = mock_use_case_with_tracker

        dto = use_case.execute(sensor_id=1, window_size=10)

        # Verificar parámetros
        param_dict = dict(tracker.params_logged)
        assert "engine_name" in param_dict
        assert param_dict["engine_name"] == "test_engine"
        assert "series_id" in param_dict
        assert "window_size" in param_dict
        assert param_dict["window_size"] == 10

    def test_tags_set(self, mock_use_case_with_tracker):
        """Los tags (pipeline_version, sensor_id) se setean."""
        use_case, tracker = mock_use_case_with_tracker

        dto = use_case.execute(sensor_id=1, window_size=10)

        # Verificar tags
        tag_dict = dict(tracker.tags_set)
        assert "pipeline_version" in tag_dict
        assert tag_dict["pipeline_version"] == "0.2.1-GOLD"
        assert "sensor_id" in tag_dict
        assert tag_dict["sensor_id"] == "1"

    def test_step_increment(self, mock_use_case_with_tracker):
        """El step incrementa con cada predicción."""
        use_case, tracker = mock_use_case_with_tracker

        # Ejecutar 3 predicciones
        use_case.execute(sensor_id=1, window_size=10)
        use_case.execute(sensor_id=2, window_size=10)
        use_case.execute(sensor_id=3, window_size=10)

        # Verificar que steps son 1, 2, 3
        confidence_metrics = [m for m in tracker.metrics_logged if m[0] == "confidence_score"]
        assert len(confidence_metrics) == 3
        steps = [m[2] for m in confidence_metrics]
        assert steps == [1, 2, 3]

    def test_null_tracker_by_default(self):
        """Sin tracker explícito, usa NullExperimentTracker (no-op)."""
        storage = MockStoragePort()
        engine = MockPredictionPort()
        domain_service = PredictionDomainService(engines=[engine])

        # No pasar tracker
        use_case = PredictSensorValueUseCase(
            prediction_service=domain_service,
            storage=storage,
        )

        # Debe funcionar sin error
        dto = use_case.execute(sensor_id=1, window_size=10)
        assert dto is not None
        assert dto.confidence_score == 0.85

    def test_tracking_failure_is_safe(self, mock_use_case_with_tracker):
        """Si el tracker falla, la predicción sigue funcionando."""
        use_case, tracker = mock_use_case_with_tracker

        # Hacer que el tracker falle
        tracker.log_metric = MagicMock(side_effect=Exception("Tracking error"))
        tracker.log_metrics = MagicMock(side_effect=Exception("Tracking error"))

        # La predicción debe seguir funcionando
        dto = use_case.execute(sensor_id=1, window_size=10)
        assert dto is not None

    def test_confidence_interval_width_logged(self, mock_use_case_with_tracker):
        """Si hay confidence_interval, se loguea su ancho."""
        use_case, tracker = mock_use_case_with_tracker

        dto = use_case.execute(sensor_id=1, window_size=10)

        # Verificar que se logueó confidence_interval_width
        metric_keys = [m[0] for m in tracker.metrics_logged]
        assert "confidence_interval_width" in metric_keys

        # El ancho debe ser 4.0 (44.0 - 40.0)
        width_metric = [m for m in tracker.metrics_logged if m[0] == "confidence_interval_width"]
        assert len(width_metric) == 1
        assert width_metric[0][1] == 4.0


class TestBackwardCompatibility:
    """Tests de backward compatibility."""

    def test_existing_code_without_tracker(self):
        """Código existente sin tracker sigue funcionando."""
        storage = MockStoragePort()
        engine = MockPredictionPort()
        domain_service = PredictionDomainService(engines=[engine])

        # Crear use case como se hacía antes (sin tracker)
        use_case = PredictSensorValueUseCase(
            prediction_service=domain_service,
            storage=storage,
            window_size=100,
        )

        # Debe funcionar
        dto = use_case.execute(sensor_id=1)
        assert dto is not None
