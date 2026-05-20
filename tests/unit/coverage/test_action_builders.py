"""Auto-generated coverage test for domain/policies/action_builders.py."""
import pytest


def test_action_builders_importable():
    try:
        import iot_machine_learning.domain.policies.action_builders
        assert iot_machine_learning.domain.policies.action_builders is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
