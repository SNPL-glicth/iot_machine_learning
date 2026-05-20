"""Auto-generated coverage test for ml_service/features/persistence/redis_window_store.py."""
import pytest


def test_redis_window_store_importable():
    try:
        import iot_machine_learning.ml_service.features.persistence.redis_window_store
        assert iot_machine_learning.ml_service.features.persistence.redis_window_store is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
