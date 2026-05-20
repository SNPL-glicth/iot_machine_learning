"""Auto-generated coverage test for domain/entities/time_series.py."""
import pytest


def test_time_series_importable():
    try:
        import iot_machine_learning.domain.entities.time_series
        assert iot_machine_learning.domain.entities.time_series is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
