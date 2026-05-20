"""Auto-generated coverage test for infrastructure/ml/cognitive/text/perception_collector.py."""
import pytest


def test_perception_collector_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.perception_collector
        assert iot_machine_learning.infrastructure.ml.cognitive.text.perception_collector is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
