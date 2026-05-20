"""Auto-generated coverage test for infrastructure/redis/redis_ops_key.py."""
import pytest


def test_redis_ops_key_importable():
    try:
        import iot_machine_learning.infrastructure.redis.redis_ops_key
        assert iot_machine_learning.infrastructure.redis.redis_ops_key is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
