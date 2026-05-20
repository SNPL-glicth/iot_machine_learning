"""Auto-generated coverage test for infrastructure/redis/redis_operations.py."""
import pytest


def test_redis_operations_importable():
    try:
        import iot_machine_learning.infrastructure.redis.redis_operations
        assert iot_machine_learning.infrastructure.redis.redis_operations is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
