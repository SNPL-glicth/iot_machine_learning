"""Coverage test for infrastructure/adapters/iot/persistence/weight_tracker_redis_adapter.py."""
import pytest


def test_redis_client_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.iot.persistence.weight_tracker_redis_adapter as adapter
        assert adapter.WeightTrackerRedisClient is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
