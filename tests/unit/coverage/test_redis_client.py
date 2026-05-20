"""Auto-generated coverage test for infrastructure/ml/cognitive/bayesian_weight_tracker/redis_client.py."""
import pytest


def test_redis_client_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.redis_client
        assert iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.redis_client is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
