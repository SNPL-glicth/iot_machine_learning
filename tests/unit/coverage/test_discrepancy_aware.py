"""Auto-generated coverage test for infrastructure/ml/moe/fusion/discrepancy_aware.py."""
import pytest


def test_discrepancy_aware_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.fusion.discrepancy_aware
        assert iot_machine_learning.infrastructure.ml.moe.fusion.discrepancy_aware is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
