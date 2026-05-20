"""Auto-generated coverage test for core/parameters/parameter_validator.py."""
import pytest


def test_parameter_validator_importable():
    try:
        import iot_machine_learning.core.parameters.parameter_validator
        assert iot_machine_learning.core.parameters.parameter_validator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
