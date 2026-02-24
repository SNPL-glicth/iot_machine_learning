"""Tests for ContextualPlasticityTracker.

Validates:
- Error recording by context
- MAE calculation per context
- Weight calculation with inverse MAE
- Window size enforcement
- Minimum samples requirement
"""

import pytest

from iot_machine_learning.domain.entities.plasticity.plasticity_context import (
    PlasticityContext,
    RegimeType,
)
from iot_machine_learning.infrastructure.ml.cognitive.plasticity.contextual_plasticity_tracker import (
    ContextualPlasticityTracker,
)


class TestContextualPlasticityTrackerInitialization:
    """Test tracker initialization and validation."""
    
    def test_default_initialization(self) -> None:
        """Test default parameters."""
        tracker = ContextualPlasticityTracker()
        assert tracker.window_size == 50
        assert tracker.min_samples == 5
        assert tracker.epsilon == 0.1
    
    def test_custom_initialization(self) -> None:
        """Test custom parameters."""
        tracker = ContextualPlasticityTracker(
            window_size=100,
            min_samples=10,
            epsilon=0.05,
        )
        assert tracker.window_size == 100
        assert tracker.min_samples == 10
        assert tracker.epsilon == 0.05
    
    def test_invalid_window_size_raises(self) -> None:
        """Test that window_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            ContextualPlasticityTracker(window_size=0)
    
    def test_invalid_min_samples_raises(self) -> None:
        """Test that min_samples > window_size raises ValueError."""
        with pytest.raises(ValueError, match="min_samples.*cannot exceed window_size"):
            ContextualPlasticityTracker(window_size=10, min_samples=20)


class TestErrorRecording:
    """Test error recording by context."""
    
    def test_record_single_error(self) -> None:
        """Test recording a single error."""
        tracker = ContextualPlasticityTracker()
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        tracker.record_error("sensor_1", "taylor", 5.0, ctx)
        
        count = tracker.get_sample_count("sensor_1", "taylor", ctx)
        assert count == 1
    
    def test_record_multiple_errors_same_context(self) -> None:
        """Test recording multiple errors in same context."""
        tracker = ContextualPlasticityTracker()
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        for i in range(10):
            tracker.record_error("sensor_1", "taylor", float(i), ctx)
        
        count = tracker.get_sample_count("sensor_1", "taylor", ctx)
        assert count == 10
    
    def test_record_errors_different_contexts(self) -> None:
        """Test that different contexts are tracked separately."""
        tracker = ContextualPlasticityTracker()
        
        ctx_stable = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.1,
            volatility=0.2,
            time_of_day=10,
            consecutive_failures=0,
            error_magnitude=1.0,
            is_critical_zone=False,
        )
        ctx_volatile = PlasticityContext(
            regime=RegimeType.VOLATILE,
            noise_ratio=0.1,
            volatility=0.8,
            time_of_day=10,
            consecutive_failures=0,
            error_magnitude=1.0,
            is_critical_zone=False,
        )
        
        tracker.record_error("sensor_1", "taylor", 5.0, ctx_stable)
        tracker.record_error("sensor_1", "taylor", 10.0, ctx_volatile)
        
        count_stable = tracker.get_sample_count("sensor_1", "taylor", ctx_stable)
        count_volatile = tracker.get_sample_count("sensor_1", "taylor", ctx_volatile)
        
        assert count_stable == 1
        assert count_volatile == 1
    
    def test_negative_error_raises(self) -> None:
        """Test that negative errors raise ValueError."""
        tracker = ContextualPlasticityTracker()
        ctx = PlasticityContext.create_default()
        
        with pytest.raises(ValueError, match="error must be >= 0"):
            tracker.record_error("sensor_1", "taylor", -1.0, ctx)


class TestMAECalculation:
    """Test MAE calculation per context."""
    
    def test_mae_with_sufficient_samples(self) -> None:
        """Test MAE calculation with enough samples."""
        tracker = ContextualPlasticityTracker(min_samples=3)
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        # Record 5 errors: [2, 4, 6, 8, 10]
        for i in range(1, 6):
            tracker.record_error("sensor_1", "taylor", float(i * 2), ctx)
        
        mae = tracker.get_contextual_mae("sensor_1", "taylor", ctx)
        
        # MAE = (2 + 4 + 6 + 8 + 10) / 5 = 6.0
        assert mae == 6.0
    
    def test_mae_insufficient_samples_returns_none(self) -> None:
        """Test that MAE returns None with insufficient samples."""
        tracker = ContextualPlasticityTracker(min_samples=5)
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        # Record only 3 errors (need 5)
        for i in range(3):
            tracker.record_error("sensor_1", "taylor", float(i), ctx)
        
        mae = tracker.get_contextual_mae("sensor_1", "taylor", ctx)
        
        assert mae is None


class TestWeightCalculation:
    """Test contextual weight calculation."""
    
    def test_weights_with_sufficient_data(self) -> None:
        """Test weight calculation with enough data for all engines."""
        tracker = ContextualPlasticityTracker(min_samples=3, epsilon=0.1)
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        # Engine A: MAE = 2.0
        for _ in range(5):
            tracker.record_error("sensor_1", "engine_a", 2.0, ctx)
        
        # Engine B: MAE = 4.0
        for _ in range(5):
            tracker.record_error("sensor_1", "engine_b", 4.0, ctx)
        
        weights = tracker.get_contextual_weights("sensor_1", ["engine_a", "engine_b"], ctx)
        
        assert weights is not None
        # weight_a = 1/(2.0 + 0.1) = 0.476
        # weight_b = 1/(4.0 + 0.1) = 0.244
        # normalized: a = 0.476/(0.476+0.244) = 0.66, b = 0.34
        assert weights["engine_a"] > weights["engine_b"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_weights_insufficient_data_returns_none(self) -> None:
        """Test that weights return None if any engine lacks data."""
        tracker = ContextualPlasticityTracker(min_samples=5)
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        # Engine A has enough data
        for _ in range(10):
            tracker.record_error("sensor_1", "engine_a", 2.0, ctx)
        
        # Engine B has insufficient data
        for _ in range(3):
            tracker.record_error("sensor_1", "engine_b", 4.0, ctx)
        
        weights = tracker.get_contextual_weights("sensor_1", ["engine_a", "engine_b"], ctx)
        
        # Should return None because engine_b lacks data
        assert weights is None


class TestWindowSizeEnforcement:
    """Test that window size is enforced."""
    
    def test_window_size_limit(self) -> None:
        """Test that only last N errors are kept."""
        tracker = ContextualPlasticityTracker(window_size=10, min_samples=5)
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        # Record 20 errors
        for i in range(20):
            tracker.record_error("sensor_1", "taylor", float(i), ctx)
        
        count = tracker.get_sample_count("sensor_1", "taylor", ctx)
        
        # Should only keep last 10
        assert count == 10


class TestReset:
    """Test reset functionality."""
    
    def test_reset_all(self) -> None:
        """Test resetting all data."""
        tracker = ContextualPlasticityTracker()
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        tracker.record_error("sensor_1", "taylor", 5.0, ctx)
        tracker.record_error("sensor_2", "baseline", 3.0, ctx)
        
        tracker.reset()
        
        assert tracker.get_sample_count("sensor_1", "taylor", ctx) == 0
        assert tracker.get_sample_count("sensor_2", "baseline", ctx) == 0
    
    def test_reset_specific_series(self) -> None:
        """Test resetting specific series."""
        tracker = ContextualPlasticityTracker()
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        tracker.record_error("sensor_1", "taylor", 5.0, ctx)
        tracker.record_error("sensor_2", "baseline", 3.0, ctx)
        
        tracker.reset(series_id="sensor_1")
        
        assert tracker.get_sample_count("sensor_1", "taylor", ctx) == 0
        assert tracker.get_sample_count("sensor_2", "baseline", ctx) == 1
    
    def test_reset_specific_engine(self) -> None:
        """Test resetting specific engine in series."""
        tracker = ContextualPlasticityTracker()
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        tracker.record_error("sensor_1", "taylor", 5.0, ctx)
        tracker.record_error("sensor_1", "baseline", 3.0, ctx)
        
        tracker.reset(series_id="sensor_1", engine_name="taylor")
        
        assert tracker.get_sample_count("sensor_1", "taylor", ctx) == 0
        assert tracker.get_sample_count("sensor_1", "baseline", ctx) == 1


class TestContextTracking:
    """Test tracking of multiple contexts."""
    
    def test_get_all_contexts(self) -> None:
        """Test retrieving all tracked contexts."""
        tracker = ContextualPlasticityTracker()
        
        ctx1 = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.1,
            volatility=0.2,
            time_of_day=10,
            consecutive_failures=0,
            error_magnitude=1.0,
            is_critical_zone=False,
        )
        ctx2 = PlasticityContext(
            regime=RegimeType.VOLATILE,
            noise_ratio=0.1,
            volatility=0.8,
            time_of_day=14,
            consecutive_failures=0,
            error_magnitude=1.0,
            is_critical_zone=False,
        )
        
        tracker.record_error("sensor_1", "taylor", 5.0, ctx1)
        tracker.record_error("sensor_1", "taylor", 10.0, ctx2)
        
        contexts = tracker.get_all_contexts("sensor_1", "taylor")
        
        assert len(contexts) == 2
        assert ctx1.context_key in contexts
        assert ctx2.context_key in contexts
