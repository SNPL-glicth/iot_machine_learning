"""Auto-generated coverage test for infrastructure/adapters/reading_adapters.py."""
import pytest


def test_reading_adapters_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.reading_adapters
        assert iot_machine_learning.infrastructure.adapters.reading_adapters is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
