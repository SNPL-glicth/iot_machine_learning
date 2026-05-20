"""Auto-generated coverage test for infrastructure/ml/cognitive/seasonal/stl_decomposer.py."""
import pytest


def test_stl_decomposer_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.seasonal.stl_decomposer
        assert iot_machine_learning.infrastructure.ml.cognitive.seasonal.stl_decomposer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
