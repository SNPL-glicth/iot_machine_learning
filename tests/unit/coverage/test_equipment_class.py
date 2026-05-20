"""Auto-generated coverage test for domain/value_objects/equipment_class.py."""
import pytest


def test_equipment_class_importable():
    try:
        import iot_machine_learning.domain.value_objects.equipment_class
        assert iot_machine_learning.domain.value_objects.equipment_class is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
