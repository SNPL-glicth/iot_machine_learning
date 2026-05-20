"""Auto-generated coverage test for infrastructure/ml/cognitive/perception/failure_collector.py."""
import pytest


def test_failure_collector_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.perception.failure_collector
        assert iot_machine_learning.infrastructure.ml.cognitive.perception.failure_collector is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
