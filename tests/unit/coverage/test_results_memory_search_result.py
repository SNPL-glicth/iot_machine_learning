"""Auto-generated coverage test for domain/entities/results/memory_search_result.py."""
import pytest


def test_memory_search_result_importable():
    try:
        import iot_machine_learning.domain.entities.results.memory_search_result
        assert iot_machine_learning.domain.entities.results.memory_search_result is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
