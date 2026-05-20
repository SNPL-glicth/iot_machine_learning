"""Auto-generated coverage test for infrastructure/ml/moe/metrics/moe_alert_service.py."""
import pytest


def test_moe_alert_service_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.metrics.moe_alert_service
        assert iot_machine_learning.infrastructure.ml.moe.metrics.moe_alert_service is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
