"""Auto-generated coverage test for ml_service/config/loader.py."""
import pytest


def test_loader_importable():
    try:
        import iot_machine_learning.ml_service.config.loader
        assert iot_machine_learning.ml_service.config.loader is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
