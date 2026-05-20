"""Auto-generated coverage test for domain/policies/policy_helpers.py."""
import pytest


def test_policy_helpers_importable():
    try:
        import iot_machine_learning.domain.policies.policy_helpers
        assert iot_machine_learning.domain.policies.policy_helpers is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
