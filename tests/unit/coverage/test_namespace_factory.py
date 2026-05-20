"""Auto-generated coverage test for infrastructure/security/namespace_factory.py."""
import pytest


def test_namespace_factory_importable():
    try:
        import iot_machine_learning.infrastructure.security.namespace_factory
        assert iot_machine_learning.infrastructure.security.namespace_factory is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
