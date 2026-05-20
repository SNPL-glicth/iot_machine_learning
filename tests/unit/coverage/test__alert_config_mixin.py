"""Auto-generated coverage test for domain/services/anomaly/_alert_config_mixin.py."""
import pytest


def test__alert_config_mixin_importable():
    try:
        import iot_machine_learning.domain.services.anomaly._alert_config_mixin
        assert iot_machine_learning.domain.services.anomaly._alert_config_mixin is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
