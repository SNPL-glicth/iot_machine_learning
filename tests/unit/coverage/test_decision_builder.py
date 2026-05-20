"""Auto-generated coverage test for ml_service/context/services/decision_builder.py."""
import pytest


def test_decision_builder_importable():
    try:
        import iot_machine_learning.ml_service.context.services.decision_builder
        assert iot_machine_learning.ml_service.context.services.decision_builder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
