"""Tests for FASE 2 — Taylor Engine Improvements.

Tests cover:
1. Coefficient caching with TTL (fixes CRIT-1)
2. MAE/RMSE tracking for confidence (fixes CRIT-2)
3. Temporal gap detection (fixes CRIT-3)
4. Config snapshot persistence (MISS-3)
"""

import pytest
import time
from unittest.mock import Mock


# Coefficient Cache Tests
class TestTaylorCoefficientCache:
    """Test coefficient caching."""

    def test_cache_hit_returns_cached_coefficients(self):
        """Should return cached coefficients on hit."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.coefficient_cache import (
            TaylorCoefficientCache,
        )
        from iot_machine_learning.infrastructure.ml.engines.taylor.types import (
            TaylorCoefficients,
        )

        cache = TaylorCoefficientCache(ttl_seconds=300)

        coeffs = TaylorCoefficients(
            f_t=20.0,
            f_prime=0.5,
            f_double_prime=0.01,
            f_triple_prime=0.0,
            estimated_order=2,
            method="backward",
        )

        window_hash = "abc123"
        cache.put("sensor_42", coeffs, window_hash, 10, 1.0)

        # Cache hit
        cached = cache.get("sensor_42", window_hash)
        assert cached is not None
        assert cached.f_t == 20.0
        assert cached.f_prime == 0.5

    def test_cache_miss_on_expired(self):
        """Should return None on expired cache."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.coefficient_cache import (
            TaylorCoefficientCache,
        )
        from iot_machine_learning.infrastructure.ml.engines.taylor.types import (
            TaylorCoefficients,
        )

        cache = TaylorCoefficientCache(ttl_seconds=1)  # 1 second TTL

        coeffs = TaylorCoefficients(
            f_t=20.0,
            f_prime=0.5,
            f_double_prime=0.01,
            f_triple_prime=0.0,
            estimated_order=2,
            method="backward",
        )

        window_hash = "abc123"
        cache.put("sensor_42", coeffs, window_hash, 10, 1.0)

        # Wait for expiration
        time.sleep(1.1)

        # Cache miss
        cached = cache.get("sensor_42", window_hash)
        assert cached is None

    def test_cache_miss_on_different_window(self):
        """Should return None on different window hash."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.coefficient_cache import (
            TaylorCoefficientCache,
        )
        from iot_machine_learning.infrastructure.ml.engines.taylor.types import (
            TaylorCoefficients,
        )

        cache = TaylorCoefficientCache()

        coeffs = TaylorCoefficients(
            f_t=20.0,
            f_prime=0.5,
            f_double_prime=0.01,
            f_triple_prime=0.0,
            estimated_order=2,
            method="backward",
        )

        cache.put("sensor_42", coeffs, "hash1", 10, 1.0)

        # Different hash
        cached = cache.get("sensor_42", "hash2")
        assert cached is None

    def test_get_metrics_returns_hit_rate(self):
        """Should track cache metrics."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.coefficient_cache import (
            TaylorCoefficientCache,
        )
        from iot_machine_learning.infrastructure.ml.engines.taylor.types import (
            TaylorCoefficients,
        )

        cache = TaylorCoefficientCache()

        coeffs = TaylorCoefficients(
            f_t=20.0,
            f_prime=0.5,
            f_double_prime=0.01,
            f_triple_prime=0.0,
            estimated_order=2,
            method="backward",
        )

        cache.put("sensor_42", coeffs, "hash1", 10, 1.0)

        # 1 hit
        cache.get("sensor_42", "hash1")

        # 1 miss
        cache.get("sensor_42", "hash2")

        metrics = cache.get_metrics()
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["hit_rate"] == 0.5


# Performance Tracker Tests
class TestTaylorPerformanceTracker:
    """Test performance tracking."""

    def test_record_error_updates_metrics(self):
        """Should update MAE/RMSE on error recording."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.performance_tracker import (
            TaylorPerformanceTracker,
        )

        tracker = TaylorPerformanceTracker()

        tracker.record_error(predicted=20.0, actual=21.0)  # error=1.0
        tracker.record_error(predicted=22.0, actual=23.0)  # error=1.0

        metrics = tracker.get_metrics()
        assert metrics is not None
        assert metrics.mae == 1.0
        assert metrics.n_samples == 2

    def test_compute_confidence_adjustment_reduces_on_high_error(self):
        """Should reduce confidence on high error."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.performance_tracker import (
            TaylorPerformanceTracker,
        )

        tracker = TaylorPerformanceTracker()

        # Record high errors
        for _ in range(10):
            tracker.record_error(predicted=20.0, actual=25.0)  # error=5.0

        base_confidence = 0.9
        value_range = 10.0  # 5.0 error / 10.0 range = 0.5 normalized

        adjusted = tracker.compute_confidence_adjustment(base_confidence, value_range)

        # Should be significantly reduced
        assert adjusted < base_confidence
        assert adjusted < 0.5  # High error penalty

    def test_compute_confidence_adjustment_no_change_on_low_error(self):
        """Should not reduce confidence on low error."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.performance_tracker import (
            TaylorPerformanceTracker,
        )

        tracker = TaylorPerformanceTracker()

        # Record low errors
        for _ in range(10):
            tracker.record_error(predicted=20.0, actual=20.1)  # error=0.1

        base_confidence = 0.9
        value_range = 100.0  # 0.1 error / 100.0 range = 0.001 normalized

        adjusted = tracker.compute_confidence_adjustment(base_confidence, value_range)

        # Should be almost unchanged
        assert adjusted >= 0.85


# Gap Detector Tests
class TestTemporalGapDetector:
    """Test temporal gap detection."""

    def test_detect_gaps_finds_large_gaps(self):
        """Should detect gaps > 3× median Δt."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.gap_detector import (
            TemporalGapDetector,
        )

        detector = TemporalGapDetector(gap_threshold_multiplier=3.0)

        # Regular intervals: 1, 2, 3, 4, then GAP, 10, 11, 12
        timestamps = [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]

        gaps = detector.detect_gaps(timestamps)

        assert len(gaps) == 1
        assert gaps[0].gap_index == 4
        assert gaps[0].gap_size == 6.0  # 10.0 - 4.0

    def test_segment_by_gaps_splits_at_gaps(self):
        """Should segment time series at gaps."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.gap_detector import (
            TemporalGapDetector,
        )

        detector = TemporalGapDetector()

        values = [20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0]
        timestamps = [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]

        segments = detector.segment_by_gaps(values, timestamps)

        assert len(segments) == 2
        assert len(segments[0][0]) == 4  # First segment
        assert len(segments[1][0]) == 3  # Second segment

    def test_get_largest_segment_returns_longest(self):
        """Should return largest continuous segment."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.gap_detector import (
            TemporalGapDetector,
        )

        detector = TemporalGapDetector()

        values = [20.0, 21.0, 30.0, 31.0, 32.0, 33.0, 34.0]
        timestamps = [1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 14.0]

        largest_values, largest_timestamps = detector.get_largest_segment(values, timestamps)

        # Second segment is larger (5 points vs 2)
        assert len(largest_values) == 5
        assert largest_values[0] == 30.0

    def test_compute_robust_dt_excludes_gaps(self):
        """Should compute Δt excluding gaps."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.gap_detector import (
            TemporalGapDetector,
        )

        detector = TemporalGapDetector()

        # Regular Δt=1.0, but with gap of 6.0
        timestamps = [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]

        dt = detector.compute_robust_dt(timestamps)

        # Should return 1.0 (median of largest segment), not affected by gap
        assert dt == 1.0


# Taylor Engine Integration Tests
class TestTaylorEngineWithFase2:
    """Test Taylor engine with FASE 2 features."""

    def test_cache_enabled_reuses_coefficients(self):
        """Should reuse cached coefficients on same window."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.engine import (
            TaylorPredictionEngine,
        )

        engine = TaylorPredictionEngine(
            enable_cache=True,
            series_id="sensor_42",
        )

        values = [20.0, 21.0, 22.0, 23.0, 24.0]

        # First prediction
        result1 = engine.predict(values)

        # Second prediction (same window)
        result2 = engine.predict(values)

        # Should have cache hit
        assert result2.metadata.get("cache_hit") is True

    def test_performance_tracking_adjusts_confidence(self):
        """Should adjust confidence based on historical error."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.engine import (
            TaylorPredictionEngine,
        )

        engine = TaylorPredictionEngine(
            enable_tracking=True,
            series_id="sensor_42",
        )

        values = [20.0, 21.0, 22.0, 23.0, 24.0]

        # First prediction
        result1 = engine.predict(values)
        base_conf1 = result1.confidence

        # Record high errors
        for _ in range(10):
            engine.record_actual(predicted=25.0, actual=30.0)  # Large error

        # Second prediction
        result2 = engine.predict(values)

        # Confidence should be reduced
        assert result2.confidence < base_conf1
        assert "performance" in result2.metadata

    def test_gap_detection_segments_window(self):
        """Should segment window at temporal gaps."""
        from iot_machine_learning.infrastructure.ml.engines.taylor.engine import (
            TaylorPredictionEngine,
        )

        engine = TaylorPredictionEngine(
            enable_gap_detection=True,
        )

        values = [20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0]
        timestamps = [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]  # Gap at index 4

        # Should use largest segment (second one with 3 points)
        result = engine.predict(values, timestamps)

        # Prediction should succeed
        assert result.predicted_value is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
