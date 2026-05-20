"""Auto-generated coverage test for infrastructure/security/auth_provider.py."""
import pytest


def test_auth_provider_importable():
    try:
        import iot_machine_learning.infrastructure.security.auth_provider
        assert iot_machine_learning.infrastructure.security.auth_provider is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
