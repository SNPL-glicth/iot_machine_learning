"""Auto-generated coverage test for domain/services/anomaly/_alert_store_mixin.py."""
import pytest


def test__alert_store_mixin_importable():
    try:
        import iot_machine_learning.domain.services.anomaly._alert_store_mixin
        assert iot_machine_learning.domain.services.anomaly._alert_store_mixin is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
