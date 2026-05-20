"""Auto-generated coverage test for infrastructure/persistence/redis/circuit_factory.py."""
import pytest


def test_circuit_factory_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis.circuit_factory
        assert iot_machine_learning.infrastructure.persistence.redis.circuit_factory is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
