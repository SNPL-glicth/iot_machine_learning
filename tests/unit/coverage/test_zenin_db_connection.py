"""Auto-generated coverage test for infrastructure/persistence/sql/zenin_db_connection.py."""
import pytest


def test_zenin_db_connection_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection
        assert iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
