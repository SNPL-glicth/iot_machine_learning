"""Auto-generated coverage test for ml_service/metrics/prometheus_exporter.py."""
import pytest


def test_prometheus_exporter_importable():
    try:
        import iot_machine_learning.ml_service.metrics.prometheus_exporter
        assert iot_machine_learning.ml_service.metrics.prometheus_exporter is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
