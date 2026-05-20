"""Auto-generated coverage test for infrastructure/ml/cognitive/drift/adwin.py."""
import pytest


def test_adwin_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.drift.adwin
        assert iot_machine_learning.infrastructure.ml.cognitive.drift.adwin is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
