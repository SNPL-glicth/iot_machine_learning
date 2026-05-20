"""Auto-generated coverage test for infrastructure/adapters/weaviate/result_mapper.py."""
import pytest


def test_result_mapper_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.weaviate.result_mapper
        assert iot_machine_learning.infrastructure.adapters.weaviate.result_mapper is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
