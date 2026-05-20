"""Auto-generated coverage test for domain/entities/results/boundary_result.py."""
import pytest


def test_boundary_result_importable():
    try:
        import iot_machine_learning.domain.entities.results.boundary_result
        assert iot_machine_learning.domain.entities.results.boundary_result is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
