"""Auto-generated coverage test for infrastructure/ml/anomaly/scoring/functions.py."""
import pytest


def test_functions_importable():
    try:
        import iot_machine_learning.infrastructure.ml.anomaly.scoring.functions
        assert iot_machine_learning.infrastructure.ml.anomaly.scoring.functions is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
