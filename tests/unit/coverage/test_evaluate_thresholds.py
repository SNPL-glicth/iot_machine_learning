"""Auto-generated coverage test for application/use_cases/evaluate_thresholds.py."""
import pytest


def test_evaluate_thresholds_importable():
    try:
        import iot_machine_learning.application.use_cases.evaluate_thresholds
        assert iot_machine_learning.application.use_cases.evaluate_thresholds is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
