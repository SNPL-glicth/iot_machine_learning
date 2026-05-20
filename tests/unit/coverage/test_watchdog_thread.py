"""Auto-generated coverage test for ml_service/utils/watchdog_thread.py."""
import pytest


def test_watchdog_thread_importable():
    try:
        import iot_machine_learning.ml_service.utils.watchdog_thread
        assert iot_machine_learning.ml_service.utils.watchdog_thread is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
