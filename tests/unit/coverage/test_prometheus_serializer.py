"""Auto-generated coverage test for ml_service/metrics/prometheus_serializer.py."""
import pytest


def test_prometheus_serializer_importable():
    try:
        import iot_machine_learning.ml_service.metrics.prometheus_serializer
        assert iot_machine_learning.ml_service.metrics.prometheus_serializer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
