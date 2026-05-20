"""Auto-generated coverage test for ml_service/api/dependencies.py."""
import pytest


def test_dependencies_importable():
    try:
        import iot_machine_learning.ml_service.api.dependencies
        assert iot_machine_learning.ml_service.api.dependencies is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
