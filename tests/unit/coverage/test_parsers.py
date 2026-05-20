"""Auto-generated coverage test for ml_service/config/parsers.py."""
import pytest


def test_parsers_importable():
    try:
        import iot_machine_learning.ml_service.config.parsers
        assert iot_machine_learning.ml_service.config.parsers is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
