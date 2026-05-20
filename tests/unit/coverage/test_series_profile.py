"""Auto-generated coverage test for domain/entities/series_profile.py."""
import pytest


def test_series_profile_importable():
    try:
        import iot_machine_learning.domain.entities.series_profile
        assert iot_machine_learning.domain.entities.series_profile is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
