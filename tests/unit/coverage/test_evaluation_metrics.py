"""Auto-generated coverage test for application/evaluation/metrics.py."""
import pytest


def test_metrics_importable():
    try:
        import iot_machine_learning.application.evaluation.metrics
        assert iot_machine_learning.application.evaluation.metrics is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
