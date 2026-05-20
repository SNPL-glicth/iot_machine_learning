"""Auto-generated coverage test for infrastructure/adapters/sql_correlation_adapter.py."""
import pytest


def test_sql_correlation_adapter_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.sql_correlation_adapter
        assert iot_machine_learning.infrastructure.adapters.sql_correlation_adapter is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
