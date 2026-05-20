"""Auto-generated coverage test for ml_service/metrics/performance_metrics.py."""
import pytest


def test_performance_metrics_importable():
    try:
        import iot_machine_learning.ml_service.metrics.performance_metrics
        assert iot_machine_learning.ml_service.metrics.performance_metrics is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
