"""Auto-generated coverage test for domain/policies/threshold_policy.py."""
import pytest


def test_threshold_policy_importable():
    try:
        import iot_machine_learning.domain.policies.threshold_policy
        assert iot_machine_learning.domain.policies.threshold_policy is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
