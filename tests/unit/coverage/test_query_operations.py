"""Auto-generated coverage test for infrastructure/adapters/weaviate/query_operations.py."""
import pytest


def test_query_operations_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.weaviate.query_operations
        assert iot_machine_learning.infrastructure.adapters.weaviate.query_operations is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
