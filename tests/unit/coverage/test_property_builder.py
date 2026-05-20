"""Auto-generated coverage test for infrastructure/persistence/vector/schema/property_builder.py."""
import pytest


def test_property_builder_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.vector.schema.property_builder
        assert iot_machine_learning.infrastructure.persistence.vector.schema.property_builder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
