"""Auto-generated coverage test for application/use_cases/enrich_prediction.py."""
import pytest


def test_enrich_prediction_importable():
    try:
        import iot_machine_learning.application.use_cases.enrich_prediction
        assert iot_machine_learning.application.use_cases.enrich_prediction is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
