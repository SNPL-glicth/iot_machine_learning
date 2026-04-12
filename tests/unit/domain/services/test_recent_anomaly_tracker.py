"""Tests for RecentAnomalyTracker implementations."""

from __future__ import annotations

import time

import pytest

from iot_machine_learning.domain.ports import NullAnomalyTracker
from iot_machine_learning.infrastructure.adapters.inmemory import InMemoryRecentAnomalyTracker


class TestNullAnomalyTracker:
    """Test NullAnomalyTracker no-op implementation."""
    
    def test_record_anomaly_noop(self) -> None:
        """record_anomaly does nothing."""
        tracker = NullAnomalyTracker()
        tracker.record_anomaly("series_1", 0.9)  # Should not raise
    
    def test_record_normal_noop(self) -> None:
        """record_normal does nothing."""
        tracker = NullAnomalyTracker()
        tracker.record_normal("series_1")  # Should not raise
    
    def test_get_count_returns_zero(self) -> None:
        """get_count_last_n_minutes always returns 0."""
        tracker = NullAnomalyTracker()
        assert tracker.get_count_last_n_minutes("series_1", 120) == 0
    
    def test_get_consecutive_returns_zero(self) -> None:
        """get_consecutive_count always returns 0."""
        tracker = NullAnomalyTracker()
        assert tracker.get_consecutive_count("series_1") == 0
    
    def test_get_rate_returns_zero(self) -> None:
        """get_anomaly_rate always returns 0.0."""
        tracker = NullAnomalyTracker()
        assert tracker.get_anomaly_rate("series_1", 120) == 0.0


class TestInMemoryRecentAnomalyTracker:
    """Test InMemoryRecentAnomalyTracker implementation."""
    
    def test_record_and_get_count(self) -> None:
        """Can record anomalies and retrieve count."""
        tracker = InMemoryRecentAnomalyTracker()
        
        tracker.record_anomaly("series_1", 0.9)
        tracker.record_anomaly("series_1", 0.8)
        
        count = tracker.get_count_last_n_minutes("series_1", 120)
        assert count == 2
    
    def test_consecutive_increments(self) -> None:
        """Consecutive counter increments on anomalies."""
        tracker = InMemoryRecentAnomalyTracker()
        
        tracker.record_anomaly("series_1", 0.9)
        assert tracker.get_consecutive_count("series_1") == 1
        
        tracker.record_anomaly("series_1", 0.8)
        assert tracker.get_consecutive_count("series_1") == 2
    
    def test_consecutive_resets_on_normal(self) -> None:
        """Consecutive counter resets on normal prediction."""
        tracker = InMemoryRecentAnomalyTracker()
        
        tracker.record_anomaly("series_1", 0.9)
        tracker.record_anomaly("series_1", 0.8)
        assert tracker.get_consecutive_count("series_1") == 2
        
        tracker.record_normal("series_1")
        assert tracker.get_consecutive_count("series_1") == 0
    
    def test_anomaly_rate_calculation(self) -> None:
        """Anomaly rate is anomalies / total."""
        tracker = InMemoryRecentAnomalyTracker()
        
        # 2 anomalies
        tracker.record_anomaly("series_1", 0.9)
        tracker.record_anomaly("series_1", 0.8)
        # 1 normal
        tracker.record_normal("series_1")
        
        rate = tracker.get_anomaly_rate("series_1", 120)
        assert rate == pytest.approx(2.0 / 3.0, abs=0.01)
    
    def test_series_isolation(self) -> None:
        """Different series have isolated counts."""
        tracker = InMemoryRecentAnomalyTracker()
        
        tracker.record_anomaly("series_a", 0.9)
        tracker.record_anomaly("series_b", 0.8)
        
        assert tracker.get_count_last_n_minutes("series_a", 120) == 1
        assert tracker.get_count_last_n_minutes("series_b", 120) == 1
    
    def test_reset_clears_all(self) -> None:
        """Reset clears all data."""
        tracker = InMemoryRecentAnomalyTracker()
        
        tracker.record_anomaly("series_1", 0.9)
        tracker.reset()
        
        assert tracker.get_count_last_n_minutes("series_1", 120) == 0
    
    def test_reset_single_series(self) -> None:
        """Reset with series_id clears only that series."""
        tracker = InMemoryRecentAnomalyTracker()
        
        tracker.record_anomaly("series_a", 0.9)
        tracker.record_anomaly("series_b", 0.8)
        
        tracker.reset("series_a")
        
        assert tracker.get_count_last_n_minutes("series_a", 120) == 0
        assert tracker.get_count_last_n_minutes("series_b", 120) == 1


class TestAnomalyRateEdgeCases:
    """Edge cases for anomaly rate."""
    
    def test_rate_zero_when_no_data(self) -> None:
        """Rate is 0.0 when no predictions."""
        tracker = InMemoryRecentAnomalyTracker()
        assert tracker.get_anomaly_rate("unknown", 120) == 0.0
    
    def test_rate_one_when_all_anomalies(self) -> None:
        """Rate is 1.0 when all are anomalies."""
        tracker = InMemoryRecentAnomalyTracker()
        
        tracker.record_anomaly("series_1", 0.9)
        tracker.record_anomaly("series_1", 0.8)
        
        rate = tracker.get_anomaly_rate("series_1", 120)
        assert rate == 1.0
