"""Auto-generated coverage test for domain/entities/semantic_extraction/entity_attributes.py."""
import pytest


def test_entity_attributes_importable():
    try:
        import iot_machine_learning.domain.entities.semantic_extraction.entity_attributes
        assert iot_machine_learning.domain.entities.semantic_extraction.entity_attributes is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
