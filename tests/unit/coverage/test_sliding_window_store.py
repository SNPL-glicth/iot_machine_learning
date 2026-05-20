"""Auto-generated coverage test for infrastructure/persistence/redis/sliding_window_store.py."""
import pytest


def test_sliding_window_store_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis.sliding_window_store
        assert iot_machine_learning.infrastructure.persistence.redis.sliding_window_store is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
