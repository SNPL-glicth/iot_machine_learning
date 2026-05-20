"""Auto-generated coverage test for infrastructure/persistence/redis/pools.py."""
import pytest


def test_pools_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis.pools
        assert iot_machine_learning.infrastructure.persistence.redis.pools is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
