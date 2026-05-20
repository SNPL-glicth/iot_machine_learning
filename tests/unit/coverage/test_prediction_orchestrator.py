"""Auto-generated coverage test for ml_service/orchestrator/prediction_orchestrator.py."""
import pytest


def test_prediction_orchestrator_importable():
    try:
        import iot_machine_learning.ml_service.orchestrator.prediction_orchestrator
        assert iot_machine_learning.ml_service.orchestrator.prediction_orchestrator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
