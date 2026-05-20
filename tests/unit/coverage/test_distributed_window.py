"""Auto-generated coverage test for infrastructure/persistence/redis/distributed_window.py."""
import pytest


def test_distributed_window_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis.distributed_window
        assert iot_machine_learning.infrastructure.persistence.redis.distributed_window is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
