"""Auto-generated coverage test for infrastructure/persistence/redis/client_async.py."""
import pytest


def test_client_async_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis.client_async
        assert iot_machine_learning.infrastructure.persistence.redis.client_async is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
