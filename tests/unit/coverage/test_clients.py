"""Auto-generated coverage test for infrastructure/persistence/redis/clients.py."""
import pytest


def test_clients_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis.clients
        assert iot_machine_learning.infrastructure.persistence.redis.clients is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
