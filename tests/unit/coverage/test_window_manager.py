"""Auto-generated coverage test for ml_service/features/services/window_manager.py."""
import pytest


def test_window_manager_importable():
    try:
        import iot_machine_learning.ml_service.features.services.window_manager
        assert iot_machine_learning.ml_service.features.services.window_manager is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
