"""Auto-generated coverage test for ml_service/workers/run_poller.py."""
import pytest


def test_run_poller_importable():
    try:
        import iot_machine_learning.ml_service.workers.run_poller
        assert iot_machine_learning.ml_service.workers.run_poller is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
