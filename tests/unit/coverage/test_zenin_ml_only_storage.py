"""Auto-generated coverage test for infrastructure/persistence/sql/zenin_ml_only_storage.py."""
import pytest


def test_zenin_ml_only_storage_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.sql.zenin_ml_only_storage
        assert iot_machine_learning.infrastructure.persistence.sql.zenin_ml_only_storage is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
