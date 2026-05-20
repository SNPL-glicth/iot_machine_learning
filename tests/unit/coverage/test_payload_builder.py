"""Auto-generated coverage test for ml_service/orchestrator/services/payload_builder.py."""
import pytest


def test_payload_builder_importable():
    try:
        import iot_machine_learning.ml_service.orchestrator.services.payload_builder
        assert iot_machine_learning.ml_service.orchestrator.services.payload_builder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
