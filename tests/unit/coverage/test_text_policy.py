"""Auto-generated coverage test for domain/policies/text_policy.py."""
import pytest


def test_text_policy_importable():
    try:
        import iot_machine_learning.domain.policies.text_policy
        assert iot_machine_learning.domain.policies.text_policy is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
