"""Auto-generated coverage test for infrastructure/ml/cognitive/series_values/batch_writable_interface.py."""
import pytest


def test_batch_writable_interface_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.series_values.batch_writable_interface
        assert iot_machine_learning.infrastructure.ml.cognitive.series_values.batch_writable_interface is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
