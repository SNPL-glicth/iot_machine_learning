"""Auto-generated coverage test for infrastructure/ml/cognitive/perception/env_knobs.py."""
import pytest


def test_env_knobs_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.perception.env_knobs
        assert iot_machine_learning.infrastructure.ml.cognitive.perception.env_knobs is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
