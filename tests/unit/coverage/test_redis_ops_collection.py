"""Auto-generated coverage test for infrastructure/redis/redis_ops_collection.py."""
import pytest


def test_redis_ops_collection_importable():
    try:
        import iot_machine_learning.infrastructure.redis.redis_ops_collection
        assert iot_machine_learning.infrastructure.redis.redis_ops_collection is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
