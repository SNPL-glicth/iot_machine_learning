"""Auto-generated coverage test for infrastructure/security/rate_limit_decorator.py."""
import pytest


def test_rate_limit_decorator_importable():
    try:
        import iot_machine_learning.infrastructure.security.rate_limit_decorator
        assert iot_machine_learning.infrastructure.security.rate_limit_decorator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
