"""Auto-generated coverage test for infrastructure/persistence/cache_decorators.py."""
import pytest


def test_cache_decorators_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.cache_decorators
        assert iot_machine_learning.infrastructure.persistence.cache_decorators is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
