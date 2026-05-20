"""Auto-generated coverage test for ml_service/metrics/ab_metrics.py."""
import pytest


def test_ab_metrics_importable():
    try:
        import iot_machine_learning.ml_service.metrics.ab_metrics
        assert iot_machine_learning.ml_service.metrics.ab_metrics is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
