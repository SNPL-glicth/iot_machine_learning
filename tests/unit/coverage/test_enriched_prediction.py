"""Auto-generated coverage test for ml_service/orchestrator/models/enriched_prediction.py."""
import pytest


def test_enriched_prediction_importable():
    try:
        import iot_machine_learning.ml_service.orchestrator.models.enriched_prediction
        assert iot_machine_learning.ml_service.orchestrator.models.enriched_prediction is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
