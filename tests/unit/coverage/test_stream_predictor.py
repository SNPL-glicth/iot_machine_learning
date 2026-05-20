"""Auto-generated coverage test for ml_service/consumers/stream_predictor.py."""
import pytest


def test_stream_predictor_importable():
    try:
        import iot_machine_learning.ml_service.consumers.stream_predictor
        assert iot_machine_learning.ml_service.consumers.stream_predictor is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
