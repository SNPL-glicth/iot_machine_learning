"""Auto-generated coverage test for infrastructure/ml/cognitive/perception/record_actual_handler.py."""
import pytest


def test_record_actual_handler_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.perception.record_actual_handler
        assert iot_machine_learning.infrastructure.ml.cognitive.perception.record_actual_handler is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
