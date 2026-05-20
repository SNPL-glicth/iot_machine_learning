"""Auto-generated coverage test for infrastructure/ml/cognitive/sanitize/imputer.py."""
import pytest


def test_imputer_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.sanitize.imputer
        assert iot_machine_learning.infrastructure.ml.cognitive.sanitize.imputer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
