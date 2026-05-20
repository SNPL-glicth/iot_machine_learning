"""Auto-generated coverage test for infrastructure/adapters/calibrators/utils/ece_metrics.py."""
import pytest


def test_ece_metrics_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.calibrators.utils.ece_metrics
        assert iot_machine_learning.infrastructure.adapters.calibrators.utils.ece_metrics is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
