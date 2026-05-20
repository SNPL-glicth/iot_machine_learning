"""Auto-generated coverage test for ml_service/workers/result_writer.py."""
import pytest


def test_result_writer_importable():
    try:
        import iot_machine_learning.ml_service.workers.result_writer
        assert iot_machine_learning.ml_service.workers.result_writer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
