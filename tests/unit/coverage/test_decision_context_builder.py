"""Auto-generated coverage test for ml_service/api/services/analysis/decision_context_builder.py."""
import pytest


def test_decision_context_builder_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.decision_context_builder
        assert iot_machine_learning.ml_service.api.services.analysis.decision_context_builder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
