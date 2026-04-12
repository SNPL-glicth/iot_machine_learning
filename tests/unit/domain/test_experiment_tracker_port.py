"""Tests para ExperimentTrackerPort y NullExperimentTracker.

Verifica la interfaz del port y el comportamiento no-op del null tracker.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.ports.experiment_tracker_port import (
    ExperimentTrackerPort,
    NullExperimentTracker,
)


class TestNullExperimentTracker:
    """Tests de NullExperimentTracker (no-op implementation)."""

    def test_start_run_returns_dummy_id(self):
        """start_run retorna ID dummy."""
        tracker = NullExperimentTracker()
        run_id = tracker.start_run("test_run", {"tag": "value"})
        
        assert isinstance(run_id, str)
        assert "null_run" in run_id
        assert tracker._run_id == run_id

    def test_end_run_clears_run_id(self):
        """end_run limpia el run_id."""
        tracker = NullExperimentTracker()
        tracker.start_run("test")
        assert tracker._run_id is not None
        
        tracker.end_run()
        assert tracker._run_id is None

    def test_log_metric_no_op(self):
        """log_metric no hace nada."""
        tracker = NullExperimentTracker()
        # No debe lanzar excepción
        tracker.log_metric("key", 1.0, step=1)
        tracker.log_metric("key", 2.0)  # Sin step

    def test_log_metrics_batch_no_op(self):
        """log_metrics batch no hace nada."""
        tracker = NullExperimentTracker()
        tracker.log_metrics({"mae": 0.5, "rmse": 0.3}, step=1)

    def test_log_param_no_op(self):
        """log_param no hace nada."""
        tracker = NullExperimentTracker()
        tracker.log_param("engine", "taylor")

    def test_log_params_batch_no_op(self):
        """log_params batch no hace nada."""
        tracker = NullExperimentTracker()
        tracker.log_params({"regime": "stable", "window": 60})

    def test_log_artifact_no_op(self):
        """log_artifact no hace nada."""
        tracker = NullExperimentTracker()
        tracker.log_artifact("/path/to/file.txt", "artifacts/")

    def test_set_tags_no_op(self):
        """set_tags no hace nada."""
        tracker = NullExperimentTracker()
        tracker.set_tags({"version": "0.2.1", "env": "test"})

    def test_context_manager_enter_starts_run(self):
        """__enter__ inicia un run."""
        tracker = NullExperimentTracker()
        
        with tracker as t:
            assert t._run_id is not None
            assert isinstance(t._run_id, str)

    def test_context_manager_exit_ends_run_finished(self):
        """__exit__ finaliza run con FINISHED si no hay excepción."""
        tracker = NullExperimentTracker()
        
        with tracker as t:
            run_id = t._run_id
        
        # Después del contexto, run_id debe ser None
        assert tracker._run_id is None

    def test_context_manager_exit_ends_run_failed(self):
        """__exit__ finaliza run con FAILED si hay excepción."""
        tracker = NullExperimentTracker()
        
        try:
            with tracker as t:
                run_id = t._run_id
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Después de la excepción, run_id debe ser None
        assert tracker._run_id is None

    def test_context_manager_returns_self(self):
        """__enter__ retorna self para uso con 'as'."""
        tracker = NullExperimentTracker()
        
        with tracker as t:
            assert t is tracker

    def test_multiple_runs_independent(self):
        """Múltiples runs son independientes."""
        tracker = NullExperimentTracker()
        
        run_id_1 = tracker.start_run("run_1")
        tracker.end_run()
        
        run_id_2 = tracker.start_run("run_2")
        tracker.end_run()
        
        # IDs deben tener formato correcto (null_run o similar)
        assert "null_run" in run_id_1
        assert "null_run" in run_id_2
        assert isinstance(run_id_1, str)
        assert isinstance(run_id_2, str)

    def test_no_side_effects(self):
        """El tracker no-op no tiene side effects."""
        tracker = NullExperimentTracker()
        
        # Realizar muchas operaciones
        for i in range(100):
            tracker.start_run(f"run_{i}")
            tracker.log_metric("metric", float(i), step=i)
            tracker.log_param("param", f"value_{i}")
            tracker.set_tags({"iteration": i})
            tracker.end_run()
        
        # No debe haber acumulación de estado
        assert tracker._run_id is None


class TestExperimentTrackerPortInterface:
    """Tests de verificación de interfaz abstracta."""

    def test_port_has_required_abstract_methods(self):
        """El port define todos los métodos abstractos requeridos."""
        abstract_methods = ExperimentTrackerPort.__abstractmethods__
        
        required_methods = [
            "start_run",
            "end_run",
            "log_metric",
            "log_metrics",
            "log_param",
            "log_params",
            "log_artifact",
            "set_tags",
            "__enter__",
            "__exit__",
        ]
        
        for method in required_methods:
            assert method in abstract_methods, f"{method} debe ser abstracto"

    def test_null_implements_all_abstract_methods(self):
        """NullExperimentTracker implementa todos los métodos abstractos."""
        abstract_methods = ExperimentTrackerPort.__abstractmethods__
        null_methods = set(dir(NullExperimentTracker))
        
        for method in abstract_methods:
            assert method in null_methods, f"NullExperimentTracker debe implementar {method}"
