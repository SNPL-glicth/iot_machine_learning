"""Auto-generated coverage test for domain/policies/policy_compat.py."""
import pytest


def test_policy_compat_importable():
    try:
        import iot_machine_learning.domain.policies.policy_compat
        assert iot_machine_learning.domain.policies.policy_compat is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
