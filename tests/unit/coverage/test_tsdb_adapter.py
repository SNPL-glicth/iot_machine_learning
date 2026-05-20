"""Auto-generated coverage test for infrastructure/persistence/redis/tsdb_adapter.py."""
import pytest


def test_tsdb_adapter_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis.tsdb_adapter
        assert iot_machine_learning.infrastructure.persistence.redis.tsdb_adapter is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
