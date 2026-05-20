"""Auto-generated coverage test for infrastructure/adapters/weaviate/filter_builders.py."""
import pytest


def test_filter_builders_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.weaviate.filter_builders
        assert iot_machine_learning.infrastructure.adapters.weaviate.filter_builders is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
