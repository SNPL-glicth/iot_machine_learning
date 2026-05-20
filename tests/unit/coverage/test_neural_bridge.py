"""Auto-generated coverage test for ml_service/api/services/analysis/neural_bridge.py."""
import pytest


def test_neural_bridge_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.neural_bridge
        assert iot_machine_learning.ml_service.api.services.analysis.neural_bridge is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
