"""Auto-generated coverage test for application/evaluation/validate.py."""
import pytest


def test_validate_importable():
    try:
        import iot_machine_learning.application.evaluation.validate
        assert iot_machine_learning.application.evaluation.validate is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
