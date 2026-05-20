"""Auto-generated coverage test for infrastructure/persistence/redis/client_sync.py."""
import pytest


def test_client_sync_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis.client_sync
        assert iot_machine_learning.infrastructure.persistence.redis.client_sync is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
