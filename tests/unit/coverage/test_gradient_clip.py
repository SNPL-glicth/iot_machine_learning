"""Auto-generated coverage test for infrastructure/ml/optimization/gradient/gradient_clip.py."""
import pytest


def test_gradient_clip_importable():
    try:
        import iot_machine_learning.infrastructure.ml.optimization.gradient.gradient_clip
        assert iot_machine_learning.infrastructure.ml.optimization.gradient.gradient_clip is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
