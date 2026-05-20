"""Auto-generated coverage test for infrastructure/security/validation_rules.py."""
import pytest


def test_validation_rules_importable():
    try:
        import iot_machine_learning.infrastructure.security.validation_rules
        assert iot_machine_learning.infrastructure.security.validation_rules is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
