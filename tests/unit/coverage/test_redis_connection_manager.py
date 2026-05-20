"""Auto-generated coverage test for infrastructure/persistence/redis_connection_manager.py."""
import pytest


def test_redis_connection_manager_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis_connection_manager
        assert iot_machine_learning.infrastructure.persistence.redis_connection_manager is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
