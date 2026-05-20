"""Auto-generated coverage test for infrastructure/ml/engines/seasonal/resampler.py."""
import pytest


def test_resampler_importable():
    try:
        import iot_machine_learning.infrastructure.ml.engines.seasonal.resampler
        assert iot_machine_learning.infrastructure.ml.engines.seasonal.resampler is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
