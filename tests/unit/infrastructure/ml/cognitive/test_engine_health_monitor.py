"""Tests for EngineHealthMonitor.

Validates:
- State tracking per engine
- Auto-inhibition by consecutive failures
- Auto-inhibition by time without success
- Reset on success
- Health summary generation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from iot_machine_learning.domain.entities.plasticity.engine_plasticity_state import (
    EnginePlasticityState,
)
from iot_machine_learning.infrastructure.ml.cognitive.engine_health_monitor import (
    EngineHealthMonitor,
)


class TestEngineHealthMonitorInitialization:
    """Test monitor initialization and validation."""
    
    def test_default_initialization(self) -> None:
        """Test default parameters."""
        monitor = EngineHealthMonitor()
        assert monitor.failure_threshold == 10
        assert monitor.max_hours_without_success == 1.0
        assert monitor.error_tolerance == 1.0
    
    def test_custom_initialization(self) -> None:
        """Test custom parameters."""
        monitor = EngineHealthMonitor(
            failure_threshold=5,
            max_hours_without_success=2.0,
            error_tolerance=0.5,
        )
        assert monitor.failure_threshold == 5
        assert monitor.max_hours_without_success == 2.0
        assert monitor.error_tolerance == 0.5
    
    def test_invalid_failure_threshold_raises(self) -> None:
        """Test that failure_threshold < 1 raises ValueError."""
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            EngineHealthMonitor(failure_threshold=0)
    
    def test_invalid_max_hours_raises(self) -> None:
        """Test that max_hours <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_hours_without_success must be > 0"):
            EngineHealthMonitor(max_hours_without_success=0)


class TestPredictionRecording:
    """Test prediction recording and state updates."""
    
    def test_record_success(self) -> None:
        """Test recording a successful prediction."""
        monitor = EngineHealthMonitor(error_tolerance=1.0)
        
        state = monitor.record_prediction("sensor_1", "taylor", error=0.5)
        
        assert state.consecutive_successes == 1
        assert state.consecutive_failures == 0
        assert state.last_error == 0.5
        assert not state.is_inhibited
    
    def test_record_failure(self) -> None:
        """Test recording a failed prediction."""
        monitor = EngineHealthMonitor(error_tolerance=1.0)
        
        state = monitor.record_prediction("sensor_1", "taylor", error=5.0)
        
        assert state.consecutive_failures == 1
        assert state.consecutive_successes == 0
        assert state.last_error == 5.0
        assert not state.is_inhibited
    
    def test_success_resets_failure_count(self) -> None:
        """Test that success resets consecutive failure count."""
        monitor = EngineHealthMonitor(error_tolerance=1.0)
        
        # Record 3 failures
        for _ in range(3):
            monitor.record_prediction("sensor_1", "taylor", error=5.0)
        
        # Record success
        state = monitor.record_prediction("sensor_1", "taylor", error=0.5)
        
        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 1
    
    def test_negative_error_raises(self) -> None:
        """Test that negative errors raise ValueError."""
        monitor = EngineHealthMonitor()
        
        with pytest.raises(ValueError, match="error must be >= 0"):
            monitor.record_prediction("sensor_1", "taylor", error=-1.0)


class TestInhibitionByFailures:
    """Test auto-inhibition by consecutive failures."""
    
    def test_inhibition_at_threshold(self) -> None:
        """Test that engine is inhibited at failure threshold."""
        monitor = EngineHealthMonitor(failure_threshold=5)
        
        # Record 5 failures
        for i in range(5):
            state = monitor.record_prediction("sensor_1", "taylor", error=10.0)
        
        # Should be inhibited after 5th failure
        assert state.is_inhibited
        assert "Consecutive failures" in state.inhibition_reason
    
    def test_no_inhibition_below_threshold(self) -> None:
        """Test that engine is not inhibited below threshold."""
        monitor = EngineHealthMonitor(failure_threshold=5)
        
        # Record 4 failures (below threshold)
        for i in range(4):
            state = monitor.record_prediction("sensor_1", "taylor", error=10.0)
        
        # Should not be inhibited yet
        assert not state.is_inhibited
    
    def test_success_prevents_inhibition(self) -> None:
        """Test that success resets count and prevents inhibition."""
        monitor = EngineHealthMonitor(failure_threshold=5, error_tolerance=1.0)
        
        # Record 4 failures
        for _ in range(4):
            monitor.record_prediction("sensor_1", "taylor", error=10.0)
        
        # Record success (resets count)
        monitor.record_prediction("sensor_1", "taylor", error=0.5)
        
        # Record 4 more failures (still below threshold)
        for _ in range(4):
            state = monitor.record_prediction("sensor_1", "taylor", error=10.0)
        
        # Should not be inhibited
        assert not state.is_inhibited


class TestInhibitionByTimeout:
    """Test auto-inhibition by time without success."""
    
    def test_inhibition_after_timeout(self) -> None:
        """Test that engine is inhibited after timeout without success."""
        monitor = EngineHealthMonitor(
            failure_threshold=100,  # High threshold to avoid failure-based inhibition
            max_hours_without_success=1.0,
        )
        
        # Create a state with last_success_time 2 hours ago
        old_success_time = datetime.now() - timedelta(hours=2)
        initial_state = EnginePlasticityState(
            engine_name="taylor",
            series_id="sensor_1",
            consecutive_failures=1,
            consecutive_successes=0,
            last_error=5.0,
            last_success_time=old_success_time,
            last_failure_time=datetime.now(),
            is_inhibited=False,
            inhibition_reason=None,
            total_predictions=1,
            total_errors=1,
        )
        
        # Manually set the state
        monitor._states["sensor_1"] = {"taylor": initial_state}
        
        # Record another failure (should trigger timeout inhibition)
        state = monitor.record_prediction("sensor_1", "taylor", error=5.0)
        
        # Should be inhibited due to timeout (2 hours > 1.0 hour limit)
        assert state.is_inhibited
        assert "No success for" in state.inhibition_reason


class TestStateRetrieval:
    """Test state retrieval methods."""
    
    def test_get_state(self) -> None:
        """Test retrieving engine state."""
        monitor = EngineHealthMonitor()
        
        monitor.record_prediction("sensor_1", "taylor", error=5.0)
        
        state = monitor.get_state("sensor_1", "taylor")
        
        assert state is not None
        assert state.engine_name == "taylor"
        assert state.series_id == "sensor_1"
    
    def test_get_state_nonexistent_returns_none(self) -> None:
        """Test that getting nonexistent state returns None."""
        monitor = EngineHealthMonitor()
        
        state = monitor.get_state("sensor_1", "taylor")
        
        assert state is None
    
    def test_get_inhibited_engines(self) -> None:
        """Test getting list of inhibited engines."""
        monitor = EngineHealthMonitor(failure_threshold=3)
        
        # Inhibit taylor
        for _ in range(3):
            monitor.record_prediction("sensor_1", "taylor", error=10.0)
        
        # baseline is healthy
        monitor.record_prediction("sensor_1", "baseline", error=0.5)
        
        inhibited = monitor.get_inhibited_engines("sensor_1")
        
        # get_inhibited_engines now returns List[Tuple[engine_name, reason]]
        inhibited_names = [name for name, reason in inhibited]
        assert "taylor" in inhibited_names
        assert "baseline" not in inhibited_names
    
    def test_is_inhibited(self) -> None:
        """Test checking if specific engine is inhibited."""
        monitor = EngineHealthMonitor(failure_threshold=3)
        
        # Inhibit taylor
        for _ in range(3):
            monitor.record_prediction("sensor_1", "taylor", error=10.0)
        
        assert monitor.is_inhibited("sensor_1", "taylor")
        assert not monitor.is_inhibited("sensor_1", "baseline")


class TestReset:
    """Test reset functionality."""
    
    def test_reset_all(self) -> None:
        """Test resetting all states."""
        monitor = EngineHealthMonitor()
        
        monitor.record_prediction("sensor_1", "taylor", error=5.0)
        monitor.record_prediction("sensor_2", "baseline", error=3.0)
        
        monitor.reset()
        
        assert monitor.get_state("sensor_1", "taylor") is None
        assert monitor.get_state("sensor_2", "baseline") is None
    
    def test_reset_specific_series(self) -> None:
        """Test resetting specific series."""
        monitor = EngineHealthMonitor()
        
        monitor.record_prediction("sensor_1", "taylor", error=5.0)
        monitor.record_prediction("sensor_2", "baseline", error=3.0)
        
        monitor.reset(series_id="sensor_1")
        
        assert monitor.get_state("sensor_1", "taylor") is None
        assert monitor.get_state("sensor_2", "baseline") is not None
    
    def test_reset_specific_engine(self) -> None:
        """Test resetting specific engine in series."""
        monitor = EngineHealthMonitor()
        
        monitor.record_prediction("sensor_1", "taylor", error=5.0)
        monitor.record_prediction("sensor_1", "baseline", error=3.0)
        
        monitor.reset(series_id="sensor_1", engine_name="taylor")
        
        assert monitor.get_state("sensor_1", "taylor") is None
        assert monitor.get_state("sensor_1", "baseline") is not None


class TestHealthSummary:
    """Test health summary generation."""
    
    def test_get_health_summary(self) -> None:
        """Test getting health summary for all engines."""
        monitor = EngineHealthMonitor(failure_threshold=3)
        
        # taylor: 3 failures (inhibited)
        for _ in range(3):
            monitor.record_prediction("sensor_1", "taylor", error=10.0)
        
        # baseline: 2 successes
        for _ in range(2):
            monitor.record_prediction("sensor_1", "baseline", error=0.5)
        
        summary = monitor.get_health_summary("sensor_1")
        
        assert "taylor" in summary
        assert "baseline" in summary
        assert summary["taylor"]["is_inhibited"] is True
        assert summary["baseline"]["is_inhibited"] is False
        assert summary["taylor"]["consecutive_failures"] == 3
        assert summary["baseline"]["consecutive_successes"] == 2
