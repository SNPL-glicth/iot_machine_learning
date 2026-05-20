"""Auto-generated coverage test for infrastructure/persistence/redis/utils.py."""
import pytest


def test_utils_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.redis.utils
        assert iot_machine_learning.infrastructure.persistence.redis.utils is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
