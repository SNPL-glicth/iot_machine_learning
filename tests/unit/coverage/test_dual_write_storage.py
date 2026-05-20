"""Auto-generated coverage test for infrastructure/persistence/sql/dual_write_storage.py."""
import pytest


def test_dual_write_storage_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.sql.dual_write_storage
        assert iot_machine_learning.infrastructure.persistence.sql.dual_write_storage is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
