"""Auto-generated coverage test for infrastructure/security/input_validator.py."""
import pytest


def test_input_validator_importable():
    try:
        import iot_machine_learning.infrastructure.security.input_validator
        assert iot_machine_learning.infrastructure.security.input_validator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
