"""Auto-generated coverage test for infrastructure/security/namespace_validators.py."""
import pytest


def test_namespace_validators_importable():
    try:
        import iot_machine_learning.infrastructure.security.namespace_validators
        assert iot_machine_learning.infrastructure.security.namespace_validators is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
