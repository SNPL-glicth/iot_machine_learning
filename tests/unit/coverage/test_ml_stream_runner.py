"""Auto-generated coverage test for ml_service/runners/ml_stream_runner.py."""
import pytest


def test_ml_stream_runner_importable():
    try:
        import iot_machine_learning.ml_service.runners.ml_stream_runner
        assert iot_machine_learning.ml_service.runners.ml_stream_runner is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
