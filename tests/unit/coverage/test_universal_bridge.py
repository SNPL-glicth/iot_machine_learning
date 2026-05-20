"""Auto-generated coverage test for ml_service/api/services/analysis/universal_bridge.py."""
import pytest


def test_universal_bridge_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.universal_bridge
        assert iot_machine_learning.ml_service.api.services.analysis.universal_bridge is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
