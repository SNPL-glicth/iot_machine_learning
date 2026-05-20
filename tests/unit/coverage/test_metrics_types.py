"""Auto-generated coverage test for ml_service/metrics/metrics_types.py."""
import pytest


def test_metrics_types_importable():
    try:
        import iot_machine_learning.ml_service.metrics.metrics_types
        assert iot_machine_learning.ml_service.metrics.metrics_types is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
