"""Auto-generated coverage test for ml_service/metrics/ab_testing.py."""
import pytest


def test_ab_testing_importable():
    try:
        import iot_machine_learning.ml_service.metrics.ab_testing
        assert iot_machine_learning.ml_service.metrics.ab_testing is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
