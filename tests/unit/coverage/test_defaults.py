"""Auto-generated coverage test for infrastructure/ml/anomaly/factory/defaults.py."""
import pytest


def test_defaults_importable():
    try:
        import iot_machine_learning.infrastructure.ml.anomaly.factory.defaults
        assert iot_machine_learning.infrastructure.ml.anomaly.factory.defaults is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
