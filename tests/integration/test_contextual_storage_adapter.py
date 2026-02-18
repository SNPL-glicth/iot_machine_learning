"""Integration tests for contextual storage adapter methods (FASE 6).

Tests the 4 new methods:
- record_contextual_error()
- get_contextual_performance()
- update_engine_health()
- get_engine_health()

These tests require SQL Server connection and migration 022 to be executed.
"""

import pytest
from datetime import datetime

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestRecordContextualError:
    """Test contextual error recording."""
    
    def test_record_contextual_error_basic(self, storage_adapter) -> None:
        """Test recording a contextual error."""
        storage_adapter.record_contextual_error(
            series_id="test_sensor_1",
            engine_name="taylor",
            predicted_value=50.0,
            actual_value=55.0,
            error=5.0,
            penalty=5.0,
            regime="STABLE",
            noise_ratio=0.1,
            volatility=0.2,
            time_of_day=14,
            consecutive_failures=0,
            is_critical_zone=False,
            context_key="stable|14|low",
        )
        
        # Verify by querying contextual performance
        perf = storage_adapter.get_contextual_performance(
            series_id="test_sensor_1",
            engine_name="taylor",
            context_key="stable|14|low",
        )
        
        # Should return None (need 5 samples minimum)
        assert perf is None
    
    def test_record_multiple_contextual_errors(self, storage_adapter) -> None:
        """Test recording multiple errors builds up context."""
        # Record 10 errors in same context
        for i in range(10):
            storage_adapter.record_contextual_error(
                series_id="test_sensor_2",
                engine_name="baseline",
                predicted_value=50.0 + i,
                actual_value=52.0 + i,
                error=2.0,
                penalty=2.0,
                regime="VOLATILE",
                noise_ratio=0.3,
                volatility=0.8,
                time_of_day=10,
                consecutive_failures=0,
                is_critical_zone=False,
                context_key="volatile|10|high",
            )
        
        # Now should have enough samples
        perf = storage_adapter.get_contextual_performance(
            series_id="test_sensor_2",
            engine_name="baseline",
            context_key="volatile|10|high",
        )
        
        assert perf is not None
        assert perf["mae"] == pytest.approx(2.0, abs=0.1)
        assert perf["count"] == 10


class TestGetContextualPerformance:
    """Test contextual performance retrieval."""
    
    def test_get_contextual_performance_insufficient_data(self, storage_adapter) -> None:
        """Test that insufficient data returns None."""
        # Record only 3 errors (need 5 minimum)
        for i in range(3):
            storage_adapter.record_contextual_error(
                series_id="test_sensor_3",
                engine_name="taylor",
                predicted_value=50.0,
                actual_value=52.0,
                error=2.0,
                penalty=2.0,
                regime="STABLE",
                noise_ratio=0.1,
                volatility=0.2,
                time_of_day=12,
                consecutive_failures=0,
                is_critical_zone=False,
                context_key="stable|12|low",
            )
        
        perf = storage_adapter.get_contextual_performance(
            series_id="test_sensor_3",
            engine_name="taylor",
            context_key="stable|12|low",
        )
        
        assert perf is None
    
    def test_get_contextual_performance_with_window(self, storage_adapter) -> None:
        """Test that window size is respected."""
        # Record 60 errors
        for i in range(60):
            storage_adapter.record_contextual_error(
                series_id="test_sensor_4",
                engine_name="taylor",
                predicted_value=50.0,
                actual_value=52.0,
                error=2.0,
                penalty=2.0,
                regime="STABLE",
                noise_ratio=0.1,
                volatility=0.2,
                time_of_day=15,
                consecutive_failures=0,
                is_critical_zone=False,
                context_key="stable|15|low",
            )
        
        # Request with window_size=50
        perf = storage_adapter.get_contextual_performance(
            series_id="test_sensor_4",
            engine_name="taylor",
            context_key="stable|15|low",
            window_size=50,
        )
        
        assert perf is not None
        assert perf["count"] == 50  # Should only return last 50


class TestUpdateEngineHealth:
    """Test engine health status updates."""
    
    def test_update_engine_health_insert(self, storage_adapter) -> None:
        """Test inserting new engine health record."""
        storage_adapter.update_engine_health(
            series_id="test_sensor_5",
            engine_name="taylor",
            consecutive_failures=3,
            consecutive_successes=0,
            total_predictions=10,
            total_errors=3,
            last_error=5.0,
            failure_rate=0.3,
            is_inhibited=False,
            inhibition_reason=None,
            last_success_time=datetime.now().isoformat(),
            last_failure_time=datetime.now().isoformat(),
        )
        
        # Verify insertion
        health = storage_adapter.get_engine_health(
            series_id="test_sensor_5",
            engine_name="taylor",
        )
        
        assert health is not None
        assert health["consecutive_failures"] == 3
        assert health["total_predictions"] == 10
        assert health["is_inhibited"] is False
    
    def test_update_engine_health_upsert(self, storage_adapter) -> None:
        """Test updating existing engine health record."""
        # Insert initial record
        storage_adapter.update_engine_health(
            series_id="test_sensor_6",
            engine_name="baseline",
            consecutive_failures=0,
            consecutive_successes=5,
            total_predictions=5,
            total_errors=0,
            last_error=0.5,
            failure_rate=0.0,
            is_inhibited=False,
        )
        
        # Update with more failures
        storage_adapter.update_engine_health(
            series_id="test_sensor_6",
            engine_name="baseline",
            consecutive_failures=10,
            consecutive_successes=0,
            total_predictions=15,
            total_errors=10,
            last_error=10.0,
            failure_rate=0.67,
            is_inhibited=True,
            inhibition_reason="Consecutive failures: 10 >= 10",
        )
        
        # Verify update
        health = storage_adapter.get_engine_health(
            series_id="test_sensor_6",
            engine_name="baseline",
        )
        
        assert health is not None
        assert health["consecutive_failures"] == 10
        assert health["total_predictions"] == 15
        assert health["is_inhibited"] is True
        assert "Consecutive failures" in health["inhibition_reason"]
    
    def test_update_engine_health_inhibition_timestamp(self, storage_adapter) -> None:
        """Test that inhibited_at is set when inhibition occurs."""
        # Insert healthy record
        storage_adapter.update_engine_health(
            series_id="test_sensor_7",
            engine_name="taylor",
            consecutive_failures=0,
            consecutive_successes=5,
            total_predictions=5,
            total_errors=0,
            last_error=0.5,
            failure_rate=0.0,
            is_inhibited=False,
        )
        
        # Update to inhibited
        storage_adapter.update_engine_health(
            series_id="test_sensor_7",
            engine_name="taylor",
            consecutive_failures=10,
            consecutive_successes=0,
            total_predictions=15,
            total_errors=10,
            last_error=10.0,
            failure_rate=0.67,
            is_inhibited=True,
            inhibition_reason="Test inhibition",
        )
        
        # Verify inhibited_at is set
        health = storage_adapter.get_engine_health(
            series_id="test_sensor_7",
            engine_name="taylor",
        )
        
        assert health is not None
        assert health["is_inhibited"] is True
        assert health["inhibited_at"] is not None


class TestGetEngineHealth:
    """Test engine health retrieval."""
    
    def test_get_engine_health_nonexistent(self, storage_adapter) -> None:
        """Test that nonexistent engine returns None."""
        health = storage_adapter.get_engine_health(
            series_id="nonexistent_sensor",
            engine_name="nonexistent_engine",
        )
        
        assert health is None
    
    def test_get_engine_health_complete_data(self, storage_adapter) -> None:
        """Test retrieving complete health data."""
        now = datetime.now().isoformat()
        
        storage_adapter.update_engine_health(
            series_id="test_sensor_8",
            engine_name="taylor",
            consecutive_failures=5,
            consecutive_successes=0,
            total_predictions=20,
            total_errors=8,
            last_error=7.5,
            failure_rate=0.4,
            is_inhibited=False,
            inhibition_reason=None,
            last_success_time=now,
            last_failure_time=now,
        )
        
        health = storage_adapter.get_engine_health(
            series_id="test_sensor_8",
            engine_name="taylor",
        )
        
        assert health is not None
        assert health["consecutive_failures"] == 5
        assert health["consecutive_successes"] == 0
        assert health["total_predictions"] == 20
        assert health["total_errors"] == 8
        assert health["last_error"] == 7.5
        assert health["failure_rate"] == pytest.approx(0.4, abs=0.01)
        assert health["is_inhibited"] is False
        assert health["last_success_time"] is not None
        assert health["last_failure_time"] is not None


# Fixture for storage adapter (assumes SQL Server connection is available)
@pytest.fixture
def storage_adapter():
    """Create storage adapter for testing.
    
    Note: This fixture assumes:
    - SQL Server is running on localhost:1434
    - Database 'IotSystem' exists
    - Migration 022 has been executed
    - User 'sa' with password 'Sandevistan2510'
    """
    from sqlalchemy import create_engine
    from iot_machine_learning.infrastructure.persistence.sql.storage import (
        SqlServerStorageAdapter,
    )
    
    connection_string = (
        "mssql+pymssql://sa:Sandevistan2510@localhost:1434/IotSystem"
    )
    
    try:
        engine = create_engine(connection_string)
        conn = engine.connect()
        adapter = SqlServerStorageAdapter(conn)
        
        yield adapter
        
        # Cleanup: delete test data
        conn.execute(text("""
            DELETE FROM contextual_prediction_errors 
            WHERE series_id LIKE 'test_sensor_%'
        """))
        conn.execute(text("""
            DELETE FROM engine_health_status 
            WHERE series_id LIKE 'test_sensor_%'
        """))
        conn.commit()
        conn.close()
        
    except Exception as e:
        pytest.skip(f"SQL Server not available: {e}")
