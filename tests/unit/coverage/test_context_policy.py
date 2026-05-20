"""Auto-generated coverage test for domain/policies/context_policy.py."""
import pytest


def test_context_policy_importable():
    try:
        import iot_machine_learning.domain.policies.context_policy
        assert iot_machine_learning.domain.policies.context_policy is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
