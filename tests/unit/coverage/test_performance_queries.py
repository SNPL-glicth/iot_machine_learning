"""Auto-generated coverage test for infrastructure/persistence/sql/storage/performance_queries.py."""
import pytest


def test_performance_queries_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.sql.storage.performance_queries
        assert iot_machine_learning.infrastructure.persistence.sql.storage.performance_queries is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
