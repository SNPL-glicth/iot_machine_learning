"""Tests for ErrorHistoryManager with Redis and memory backends."""

from __future__ import annotations

import pytest
import os

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.error_history_manager import (
    ErrorHistoryManager,
    create_error_history_manager,
)


class TestMemoryBackend:
    """Test suite for memory backend (default)."""
    
    def test_default_initialization(self) -> None:
        """Default backend is memory."""
        manager = ErrorHistoryManager()
        assert manager.backend == "memory"
    
    def test_record_and_get_errors(self) -> None:
        """Can record and retrieve errors."""
        manager = ErrorHistoryManager()
        manager.record_error("series_1", "taylor", 5.0)
        manager.record_error("series_1", "taylor", 3.0)
        
        errors = manager.get_errors("series_1", "taylor")
        assert len(errors) == 2
        assert errors[0] == 5.0
        assert errors[1] == 3.0
    
    def test_series_isolation(self) -> None:
        """Different series have isolated histories."""
        manager = ErrorHistoryManager()
        manager.record_error("series_a", "taylor", 5.0)
        manager.record_error("series_b", "taylor", 3.0)
        
        errors_a = manager.get_errors("series_a", "taylor")
        errors_b = manager.get_errors("series_b", "taylor")
        
        assert errors_a == [5.0]
        assert errors_b == [3.0]
    
    def test_engine_isolation(self) -> None:
        """Different engines have isolated histories."""
        manager = ErrorHistoryManager()
        manager.record_error("series_1", "taylor", 5.0)
        manager.record_error("series_1", "baseline", 3.0)
        
        assert manager.get_errors("series_1", "taylor") == [5.0]
        assert manager.get_errors("series_1", "baseline") == [3.0]
    
    def test_error_dict_for_inhibition(self) -> None:
        """Get errors formatted for inhibition gate."""
        manager = ErrorHistoryManager()
        manager.record_error("series_1", "taylor", 5.0)
        manager.record_error("series_1", "baseline", 3.0)
        
        result = manager.get_error_dict_for_inhibition("series_1", ["taylor", "baseline"])
        
        assert result["taylor"] == [5.0]
        assert result["baseline"] == [3.0]
    
    def test_get_all_errors_for_series(self) -> None:
        """Get all errors for a series."""
        manager = ErrorHistoryManager()
        manager.record_error("series_1", "taylor", 5.0)
        manager.record_error("series_1", "baseline", 3.0)
        
        result = manager.get_all_errors_for_series("series_1")
        
        assert result["taylor"] == [5.0]
        assert result["baseline"] == [3.0]
    
    def test_reset_all(self) -> None:
        """Reset all history."""
        manager = ErrorHistoryManager()
        manager.record_error("series_1", "taylor", 5.0)
        manager.reset()
        
        assert manager.get_errors("series_1", "taylor") == []
    
    def test_reset_series(self) -> None:
        """Reset specific series."""
        manager = ErrorHistoryManager()
        manager.record_error("series_a", "taylor", 5.0)
        manager.record_error("series_b", "taylor", 3.0)
        manager.reset("series_a")
        
        assert manager.get_errors("series_a", "taylor") == []
        assert manager.get_errors("series_b", "taylor") == [3.0]
    
    def test_reset_engine(self) -> None:
        """Reset specific engine."""
        manager = ErrorHistoryManager()
        manager.record_error("series_1", "taylor", 5.0)
        manager.record_error("series_1", "baseline", 3.0)
        manager.reset("series_1", "taylor")
        
        assert manager.get_errors("series_1", "taylor") == []
        assert manager.get_errors("series_1", "baseline") == [3.0]
    
    def test_get_all_series_ids(self) -> None:
        """Get all series IDs (for gossip protocol)."""
        manager = ErrorHistoryManager()
        manager.record_error("series_a", "taylor", 5.0)
        manager.record_error("series_b", "taylor", 3.0)
        
        series_ids = manager.get_all_series_ids()
        
        assert "series_a" in series_ids
        assert "series_b" in series_ids
    
    def test_negative_error_raises(self) -> None:
        """Negative error raises ValueError."""
        manager = ErrorHistoryManager()
        with pytest.raises(ValueError, match="error >= 0 required"):
            manager.record_error("series_1", "taylor", -5.0)


class TestMaxHistory:
    """Test max_history limit."""
    
    def test_respects_max_history(self) -> None:
        """Only keeps max_history errors."""
        manager = ErrorHistoryManager(max_history=3)
        
        for i in range(5):
            manager.record_error("series_1", "taylor", float(i))
        
        errors = manager.get_errors("series_1", "taylor")
        assert len(errors) == 3
        assert errors == [2.0, 3.0, 4.0]  # Oldest evicted


class TestRedisBackendFallback:
    """Test Redis backend with fallback to memory."""
    
    def test_redis_backend_no_client_fallback(self) -> None:
        """Redis backend without client falls back to memory."""
        manager = ErrorHistoryManager(backend="redis")
        # Falls back to memory
        assert manager.backend == "memory"
    
    def test_factory_function(self) -> None:
        """Factory creates manager with correct defaults."""
        manager = create_error_history_manager()
        assert manager.backend == "memory"


class TestEnvironmentVariable:
    """Test ML_ERROR_HISTORY_BACKEND environment variable."""
    
    def test_backend_from_env(self, monkeypatch) -> None:
        """Backend can be set via environment variable."""
        monkeypatch.setenv("ML_ERROR_HISTORY_BACKEND", "redis")
        manager = ErrorHistoryManager()
        # Falls back to memory because no Redis client
        assert manager.backend == "memory"


class TestRecordErrorsFromPerceptions:
    """Test batch error recording."""
    
    def test_record_from_perceptions(self) -> None:
        """Record errors from perceptions."""
        class MockPerception:
            def __init__(self, engine_name: str, predicted_value: float):
                self.engine_name = engine_name
                self.predicted_value = predicted_value
        
        manager = ErrorHistoryManager()
        perceptions = [
            MockPerception("taylor", 10.0),
            MockPerception("baseline", 12.0),
        ]
        
        manager.record_errors_from_perceptions("series_1", perceptions, actual_value=15.0)
        
        assert manager.get_errors("series_1", "taylor") == [5.0]
        assert manager.get_errors("series_1", "baseline") == [3.0]
