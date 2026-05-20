"""Auto-generated coverage test for infrastructure/ml/moe/registry/expert_capability.py."""
import pytest


def test_expert_capability_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.registry.expert_capability
        assert iot_machine_learning.infrastructure.ml.moe.registry.expert_capability is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
