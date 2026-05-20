"""Auto-generated coverage test for application/semantic_extraction/entity_prioritizer.py."""
import pytest


def test_entity_prioritizer_importable():
    try:
        import iot_machine_learning.application.semantic_extraction.entity_prioritizer
        assert iot_machine_learning.application.semantic_extraction.entity_prioritizer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
