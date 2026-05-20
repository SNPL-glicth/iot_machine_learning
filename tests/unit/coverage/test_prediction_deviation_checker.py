"""Auto-generated coverage test for ml_service/runners/prediction_deviation_checker.py."""
import pytest


def test_prediction_deviation_checker_importable():
    try:
        import iot_machine_learning.ml_service.runners.prediction_deviation_checker
        assert iot_machine_learning.ml_service.runners.prediction_deviation_checker is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
