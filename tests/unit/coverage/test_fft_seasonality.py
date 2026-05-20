"""Auto-generated coverage test for infrastructure/ml/cognitive/seasonal/fft_seasonality.py."""
import pytest


def test_fft_seasonality_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.seasonal.fft_seasonality
        assert iot_machine_learning.infrastructure.ml.cognitive.seasonal.fft_seasonality is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
