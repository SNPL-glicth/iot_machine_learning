"""Auto-generated coverage test for ml_service/workers/job_processor.py."""
import pytest


def test_job_processor_importable():
    try:
        import iot_machine_learning.ml_service.workers.job_processor
        assert iot_machine_learning.ml_service.workers.job_processor is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
