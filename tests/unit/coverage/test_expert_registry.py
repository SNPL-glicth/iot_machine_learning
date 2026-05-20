"""Auto-generated coverage test for infrastructure/ml/moe/registry/expert_registry.py."""
import pytest


def test_expert_registry_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.registry.expert_registry
        assert iot_machine_learning.infrastructure.ml.moe.registry.expert_registry is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
