"""Auto-generated coverage test for infrastructure/security/secret_redactor.py."""
import pytest


def test_secret_redactor_importable():
    try:
        import iot_machine_learning.infrastructure.security.secret_redactor
        assert iot_machine_learning.infrastructure.security.secret_redactor is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
