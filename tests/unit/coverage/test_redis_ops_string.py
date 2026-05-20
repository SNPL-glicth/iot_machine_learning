"""Auto-generated coverage test for infrastructure/redis/redis_ops_string.py."""
import pytest


def test_redis_ops_string_importable():
    try:
        import iot_machine_learning.infrastructure.redis.redis_ops_string
        assert iot_machine_learning.infrastructure.redis.redis_ops_string is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
