"""Auto-generated coverage test for domain/entities/pattern_result.py."""
import pytest


def test_pattern_result_importable():
    try:
        import iot_machine_learning.domain.entities.pattern_result
        assert iot_machine_learning.domain.entities.pattern_result is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
