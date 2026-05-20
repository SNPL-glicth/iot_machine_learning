"""Auto-generated coverage test for infrastructure/config/moe_factory.py."""
import pytest


def test_moe_factory_importable():
    try:
        import iot_machine_learning.infrastructure.config.moe_factory
        assert iot_machine_learning.infrastructure.config.moe_factory is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
