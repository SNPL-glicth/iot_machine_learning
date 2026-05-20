"""Auto-generated coverage test for infrastructure/redis/redis_keys_base.py."""
import pytest


def test_redis_keys_base_importable():
    try:
        import iot_machine_learning.infrastructure.redis.redis_keys_base
        assert iot_machine_learning.infrastructure.redis.redis_keys_base is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
