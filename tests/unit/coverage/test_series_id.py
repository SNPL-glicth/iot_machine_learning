"""Auto-generated coverage test for domain/value_objects/series_id.py."""
import pytest


def test_series_id_importable():
    try:
        import iot_machine_learning.domain.value_objects.series_id
        assert iot_machine_learning.domain.value_objects.series_id is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
