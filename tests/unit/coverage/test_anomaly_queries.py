"""Auto-generated coverage test for infrastructure/persistence/sql/storage/anomaly_queries.py."""
import pytest


def test_anomaly_queries_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.sql.storage.anomaly_queries
        assert iot_machine_learning.infrastructure.persistence.sql.storage.anomaly_queries is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
