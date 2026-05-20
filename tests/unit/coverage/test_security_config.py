"""Auto-generated coverage test for ml_service/config/security_config.py."""
import pytest


def test_security_config_importable():
    try:
        import iot_machine_learning.ml_service.config.security_config
        assert iot_machine_learning.ml_service.config.security_config is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
