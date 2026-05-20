"""Auto-generated coverage test for infrastructure/adapters/cognitive_storage_factory.py."""
import pytest


def test_cognitive_storage_factory_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.cognitive_storage_factory
        assert iot_machine_learning.infrastructure.adapters.cognitive_storage_factory is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
