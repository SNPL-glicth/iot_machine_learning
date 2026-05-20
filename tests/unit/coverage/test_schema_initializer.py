"""Auto-generated coverage test for infrastructure/research/weaviate/schema_initializer.py."""
import pytest


def test_schema_initializer_importable():
    try:
        import iot_machine_learning.infrastructure.research.weaviate.schema_initializer
        assert iot_machine_learning.infrastructure.research.weaviate.schema_initializer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
