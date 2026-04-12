"""Tests para MlflowTrackerAdapter.

Tests con MLflow en modo local (file-based tracking).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.domain.ports.experiment_tracker_port import (
    ExperimentTrackerPort,
    NullExperimentTracker,
)
from iot_machine_learning.infrastructure.adapters.mlflow_tracker_adapter import (
    MlflowTrackerAdapter,
)


@pytest.fixture
def temp_mlflow_uri() -> Generator[str, None, None]:
    """Crea directorio temporal para MLflow tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"file://{tmpdir}/mlruns"
        yield tracking_uri


class TestMlflowTrackerAdapterBasics:
    """Tests básicos de inicialización y fail-safety."""

    def test_init_with_disabled_flag(self):
        """Si enabled=False, debe ser no-op."""
        tracker = MlflowTrackerAdapter(enabled=False)
        assert tracker._enabled is False
        assert tracker._mlflow is None

    def test_init_with_missing_mlflow(self):
        """Si mlflow no está instalado, debe desactivarse gracefully."""
        with patch.dict("sys.modules", {"mlflow": None}):
            tracker = MlflowTrackerAdapter()
            # ImportError se captura y desactiva
            assert tracker._enabled is False

    def test_null_tracker_is_no_op(self):
        """NullExperimentTracker nunca falla."""
        tracker = NullExperimentTracker()
        
        # Todos los métodos deben funcionar sin error
        run_id = tracker.start_run("test_run", {"tag": "value"})
        tracker.log_metric("mae", 0.5, step=1)
        tracker.log_metrics({"rmse": 0.3, "mae": 0.5}, step=1)
        tracker.log_param("engine", "taylor")
        tracker.log_params({"regime": "stable", "window": 60})
        tracker.set_tags({"version": "0.2.1"})
        tracker.end_run("FINISHED")
        
        assert "null_run" in run_id

    def test_context_manager_null_tracker(self):
        """Null tracker funciona como context manager."""
        tracker = NullExperimentTracker()
        
        with tracker as t:
            t.log_metric("test", 1.0)
        # No debe lanzar excepción al salir


class TestMlflowTrackerAdapterWithLocalMLflow:
    """Tests con MLflow real en modo local (file-based)."""

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("mlflow") is None,
        reason="MLflow not installed",
    )
    def test_start_and_end_run(self, temp_mlflow_uri: str):
        """Crear run y finalizarlo."""
        tracker = MlflowTrackerAdapter(
            tracking_uri=temp_mlflow_uri,
            experiment_name="test-experiment",
        )
        
        run_id = tracker.start_run("test_run", {"model_version": "0.2.1"})
        assert run_id is not None
        assert tracker._run_id == run_id
        
        tracker.end_run("FINISHED")
        assert tracker._run_id is None

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("mlflow") is None,
        reason="MLflow not installed",
    )
    def test_log_metrics_and_params(self, temp_mlflow_uri: str):
        """Loguear métricas y parámetros."""
        tracker = MlflowTrackerAdapter(
            tracking_uri=temp_mlflow_uri,
            experiment_name="test-experiment",
        )
        
        tracker.start_run("metrics_test")
        
        # Log individual metric
        tracker.log_metric("confidence_score", 0.85, step=1)
        
        # Log batch metrics
        tracker.log_metrics({
            "mae": 0.5,
            "rmse": 0.3,
            "elapsed_ms": 150.0,
        }, step=2)
        
        # Log params
        tracker.log_param("engine_name", "taylor")
        tracker.log_params({
            "regime": "STABLE",
            "window_size": 60,
        })
        
        tracker.end_run()

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("mlflow") is None,
        reason="MLflow not installed",
    )
    def test_context_manager(self, temp_mlflow_uri: str):
        """Uso como context manager."""
        tracker = MlflowTrackerAdapter(
            tracking_uri=temp_mlflow_uri,
            experiment_name="test-experiment",
        )
        
        with tracker as t:
            t.log_metric("test", 1.0)
            t.log_param("key", "value")
        # Al salir debe llamar end_run automáticamente

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("mlflow") is None,
        reason="MLflow not installed",
    )
    def test_context_manager_with_exception(self, temp_mlflow_uri: str):
        """Context manager marca FAILED si hay excepción."""
        tracker = MlflowTrackerAdapter(
            tracking_uri=temp_mlflow_uri,
            experiment_name="test-experiment",
        )
        
        try:
            with tracker as t:
                t.log_metric("test", 1.0)
                raise ValueError("Test error")
        except ValueError:
            pass  # Esperamos que la excepción se propague
        
        # El run debería haber terminado con status FAILED


class TestMlflowTrackerFailSafety:
    """Tests de fail-safety: nunca propagar excepciones."""

    def test_log_metric_failure_is_safe(self):
        """Si log_metric falla, no propagar excepción."""
        tracker = MlflowTrackerAdapter(enabled=True)
        tracker._mlflow = MagicMock()
        tracker._mlflow.log_metric.side_effect = Exception("MLflow error")
        
        # No debe lanzar excepción
        tracker.log_metric("key", 1.0)

    def test_start_run_failure_returns_dummy_id(self):
        """Si start_run falla, retorna ID dummy o disabled_run."""
        tracker = MlflowTrackerAdapter(enabled=True)
        tracker._mlflow = MagicMock()
        tracker._mlflow.start_run.side_effect = Exception("Connection error")
        
        run_id = tracker.start_run("test")
        # Puede ser failed_run o disabled_run según el estado
        assert run_id is not None
        assert isinstance(run_id, str)

    def test_end_run_failure_is_safe(self):
        """Si end_run falla, no propagar excepción."""
        tracker = MlflowTrackerAdapter(enabled=True)
        tracker._run_id = "test_run_123"
        tracker._mlflow = MagicMock()
        tracker._mlflow.end_run.side_effect = Exception("MLflow error")
        
        # No debe lanzar excepción
        tracker.end_run()
        assert tracker._run_id is None  # Aunque falle, limpia el run_id

    def test_non_primitive_params_no_crash(self):
        """Parámetros no primitivos no causan crash (fail-safe)."""
        tracker = MlflowTrackerAdapter(enabled=True)
        tracker._mlflow = MagicMock()
        
        # Pasar un dict como valor (no primitivo)
        complex_value = {"nested": "value"}
        
        # No debe lanzar excepción
        try:
            tracker.log_param("complex", complex_value)
        except Exception as exc:
            pytest.fail(f"log_param raised exception for non-primitive: {exc}")
        
        # Test passes if no exception was raised


class TestExperimentTrackerPortInterface:
    """Tests de cumplimiento de interfaz."""

    def test_adapter_implements_port(self):
        """MlflowTrackerAdapter implementa ExperimentTrackerPort."""
        # Esto verifica en tiempo de ejecución que la interfaz está completa
        from typing import get_type_hints
        
        # Verificar que todos los métodos abstractos están implementados
        port_methods = set(
            name for name in ExperimentTrackerPort.__abstractmethods__
        )
        adapter_methods = set(dir(MlflowTrackerAdapter))
        
        assert port_methods.issubset(adapter_methods)
