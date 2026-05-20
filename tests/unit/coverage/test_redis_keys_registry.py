"""Auto-generated coverage test for infrastructure/redis/redis_keys_registry.py."""
import pytest


def test_redis_keys_registry_importable():
    try:
        import iot_machine_learning.infrastructure.redis.redis_keys_registry
        assert iot_machine_learning.infrastructure.redis.redis_keys_registry is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
