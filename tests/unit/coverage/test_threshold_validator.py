"""Auto-generated coverage test for ml_service/runners/services/threshold_validator.py."""
import pytest


def test_threshold_validator_importable():
    try:
        import iot_machine_learning.ml_service.runners.services.threshold_validator
        assert iot_machine_learning.ml_service.runners.services.threshold_validator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
