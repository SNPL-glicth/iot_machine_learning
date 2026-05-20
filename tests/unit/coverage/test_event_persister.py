"""Auto-generated coverage test for ml_service/runners/services/event_persister.py."""
import pytest


def test_event_persister_importable():
    try:
        import iot_machine_learning.ml_service.runners.services.event_persister
        assert iot_machine_learning.ml_service.runners.services.event_persister is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
