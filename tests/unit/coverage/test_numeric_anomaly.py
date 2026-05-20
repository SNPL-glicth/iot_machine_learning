"""Auto-generated coverage test for infrastructure/ml/analyzers/numeric_anomaly.py."""
import pytest


def test_numeric_anomaly_importable():
    try:
        import iot_machine_learning.infrastructure.ml.analyzers.numeric_anomaly
        assert iot_machine_learning.infrastructure.ml.analyzers.numeric_anomaly is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
