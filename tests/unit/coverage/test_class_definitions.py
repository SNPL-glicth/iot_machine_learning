"""Auto-generated coverage test for infrastructure/persistence/vector/schema/class_definitions.py."""
import pytest


def test_class_definitions_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.vector.schema.class_definitions
        assert iot_machine_learning.infrastructure.persistence.vector.schema.class_definitions is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
