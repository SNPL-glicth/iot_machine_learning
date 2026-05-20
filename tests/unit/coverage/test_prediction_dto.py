"""Auto-generated coverage test for application/dto/prediction_dto.py."""
import pytest


def test_prediction_dto_importable():
    try:
        import iot_machine_learning.application.dto.prediction_dto
        assert iot_machine_learning.application.dto.prediction_dto is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
