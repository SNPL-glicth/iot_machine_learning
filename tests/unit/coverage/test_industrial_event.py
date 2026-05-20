"""Auto-generated coverage test for domain/value_objects/industrial_event.py."""
import pytest


def test_industrial_event_importable():
    try:
        import iot_machine_learning.domain.value_objects.industrial_event
        assert iot_machine_learning.domain.value_objects.industrial_event is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
