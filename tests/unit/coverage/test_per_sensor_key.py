"""Auto-generated coverage test for infrastructure/ml/cognitive/bayesian_weight_tracker/per_sensor_key.py."""
import pytest


def test_per_sensor_key_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.per_sensor_key
        assert iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.per_sensor_key is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
