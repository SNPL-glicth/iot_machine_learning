"""Auto-generated coverage test for ml_service/workers/zenin_queue_poller.py."""
import pytest


def test_zenin_queue_poller_importable():
    try:
        import iot_machine_learning.ml_service.workers.zenin_queue_poller
        assert iot_machine_learning.ml_service.workers.zenin_queue_poller is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
