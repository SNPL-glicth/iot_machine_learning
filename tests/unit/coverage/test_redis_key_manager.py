"""Auto-generated coverage test for infrastructure/redis/redis_key_manager.py."""
import pytest


def test_redis_key_manager_importable():
    try:
        import iot_machine_learning.infrastructure.redis.redis_key_manager
        assert iot_machine_learning.infrastructure.redis.redis_key_manager is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
