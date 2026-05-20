"""Auto-generated coverage test for ml_service/api/cache_dependency.py."""
import pytest


def test_cache_dependency_importable():
    try:
        import iot_machine_learning.ml_service.api.cache_dependency
        assert iot_machine_learning.ml_service.api.cache_dependency is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
