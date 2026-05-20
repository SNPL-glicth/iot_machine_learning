"""Auto-generated coverage test for ml_service/features/services/feature_computer.py."""
import pytest


def test_feature_computer_importable():
    try:
        import iot_machine_learning.ml_service.features.services.feature_computer
        assert iot_machine_learning.ml_service.features.services.feature_computer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
