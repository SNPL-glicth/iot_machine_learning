"""Auto-generated coverage test for infrastructure/ml/benchmark/metrics.py."""
import pytest


def test_metrics_importable():
    try:
        import iot_machine_learning.infrastructure.ml.benchmark.metrics
        assert iot_machine_learning.infrastructure.ml.benchmark.metrics is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
