"""Tests for AdaptiveLearningRate calculator.

Validates:
- Logarithmic scaling by error magnitude
- Regime-based adjustments
- Noise penalty
- Failure streak boost
- Safety limits
"""

import pytest

from iot_machine_learning.domain.entities.plasticity.plasticity_context import (
    PlasticityContext,
    RegimeType,
)
from iot_machine_learning.infrastructure.ml.cognitive.adaptive_learning_rate import (
    AdaptiveLearningRate,
)


class TestAdaptiveLearningRateInitialization:
    """Test AdaptiveLearningRate initialization and validation."""
    
    def test_default_initialization(self) -> None:
        """Test default parameter initialization."""
        calc = AdaptiveLearningRate()
        assert calc.base_lr == 0.05
        assert calc.lr_min == 0.001
        assert calc.lr_max == 0.2
        assert calc.error_scale == 10.0
        assert calc.noise_penalty == 0.5
        assert calc.failure_boost == 2.0
        assert calc.failure_threshold == 5
    
    def test_custom_initialization(self) -> None:
        """Test custom parameter initialization."""
        calc = AdaptiveLearningRate(
            base_lr=0.1,
            lr_min=0.01,
            lr_max=0.5,
            error_scale=20.0,
        )
        assert calc.base_lr == 0.1
        assert calc.lr_min == 0.01
        assert calc.lr_max == 0.5
        assert calc.error_scale == 20.0
    
    def test_invalid_lr_bounds(self) -> None:
        """Test validation of lr_min < lr_max."""
        with pytest.raises(ValueError, match="0 < lr_min < lr_max"):
            AdaptiveLearningRate(lr_min=0.2, lr_max=0.1)
    
    def test_invalid_base_lr(self) -> None:
        """Test validation of base_lr <= lr_max."""
        with pytest.raises(ValueError, match="base_lr must be"):
            AdaptiveLearningRate(base_lr=0.3, lr_max=0.2)


class TestErrorMagnitudeScaling:
    """Test logarithmic scaling by error magnitude."""
    
    def test_small_error_low_lr(self) -> None:
        """Small errors should produce low learning rates."""
        calc = AdaptiveLearningRate(base_lr=0.05)
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        lr = calc.compute_adaptive_lr(error=0.1, context=ctx)
        
        # Small error should stay close to base_lr
        assert 0.04 <= lr <= 0.06
    
    def test_medium_error_medium_lr(self) -> None:
        """Medium errors should produce medium learning rates."""
        calc = AdaptiveLearningRate(base_lr=0.05)
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        lr = calc.compute_adaptive_lr(error=5.0, context=ctx)
        
        # Medium error should increase lr moderately
        assert 0.06 <= lr <= 0.12
    
    def test_large_error_high_lr(self) -> None:
        """Large errors should produce high learning rates."""
        calc = AdaptiveLearningRate(base_lr=0.05)
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        lr = calc.compute_adaptive_lr(error=50.0, context=ctx)
        
        # Large error should increase lr significantly
        assert 0.10 <= lr <= 0.2
    
    def test_logarithmic_scaling(self) -> None:
        """Learning rate should scale logarithmically, not linearly."""
        calc = AdaptiveLearningRate(base_lr=0.05)
        ctx = PlasticityContext.create_default(RegimeType.STABLE)
        
        lr_1 = calc.compute_adaptive_lr(error=1.0, context=ctx)
        lr_10 = calc.compute_adaptive_lr(error=10.0, context=ctx)
        lr_100 = calc.compute_adaptive_lr(error=100.0, context=ctx)
        
        # Logarithmic: doubling error should NOT double lr
        assert lr_10 < lr_1 * 10
        assert lr_100 < lr_10 * 10
    
    def test_negative_error_raises(self) -> None:
        """Negative errors should raise ValueError."""
        calc = AdaptiveLearningRate()
        ctx = PlasticityContext.create_default()
        
        with pytest.raises(ValueError, match="error must be >= 0"):
            calc.compute_adaptive_lr(error=-1.0, context=ctx)


class TestRegimeAdjustments:
    """Test learning rate adjustments by signal regime."""
    
    def test_stable_regime_baseline(self) -> None:
        """STABLE regime should use baseline factor (1.0)."""
        calc = AdaptiveLearningRate(base_lr=0.05)
        ctx = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.0,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        
        lr = calc.compute_adaptive_lr(error=5.0, context=ctx)
        
        # STABLE should be close to base calculation
        assert 0.05 <= lr <= 0.15
    
    def test_volatile_regime_higher_lr(self) -> None:
        """VOLATILE regime should increase learning rate (1.5x)."""
        calc = AdaptiveLearningRate(base_lr=0.05)
        
        ctx_stable = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.0,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        ctx_volatile = PlasticityContext(
            regime=RegimeType.VOLATILE,
            noise_ratio=0.0,
            volatility=0.8,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        
        lr_stable = calc.compute_adaptive_lr(error=5.0, context=ctx_stable)
        lr_volatile = calc.compute_adaptive_lr(error=5.0, context=ctx_volatile)
        
        # VOLATILE should have higher lr
        assert lr_volatile > lr_stable * 1.3
    
    def test_noisy_regime_lower_lr(self) -> None:
        """NOISY regime should decrease learning rate (0.8x)."""
        calc = AdaptiveLearningRate(base_lr=0.05)
        
        ctx_stable = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.0,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        ctx_noisy = PlasticityContext(
            regime=RegimeType.NOISY,
            noise_ratio=0.5,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        
        lr_stable = calc.compute_adaptive_lr(error=5.0, context=ctx_stable)
        lr_noisy = calc.compute_adaptive_lr(error=5.0, context=ctx_noisy)
        
        # NOISY should have lower lr
        assert lr_noisy < lr_stable * 0.9


class TestNoisePenalty:
    """Test noise ratio penalty."""
    
    def test_low_noise_no_penalty(self) -> None:
        """Low noise (< 0.3) should not trigger penalty."""
        calc = AdaptiveLearningRate(base_lr=0.05)
        
        ctx_clean = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.1,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        ctx_moderate = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.25,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        
        lr_clean = calc.compute_adaptive_lr(error=5.0, context=ctx_clean)
        lr_moderate = calc.compute_adaptive_lr(error=5.0, context=ctx_moderate)
        
        # Both should be similar (no penalty)
        assert abs(lr_clean - lr_moderate) < 0.01
    
    def test_high_noise_penalty(self) -> None:
        """High noise (> 0.3) should trigger 50% penalty."""
        calc = AdaptiveLearningRate(base_lr=0.05, noise_penalty=0.5)
        
        ctx_low_noise = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.2,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        ctx_high_noise = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.5,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        
        lr_low = calc.compute_adaptive_lr(error=5.0, context=ctx_low_noise)
        lr_high = calc.compute_adaptive_lr(error=5.0, context=ctx_high_noise)
        
        # High noise should reduce lr by ~50%
        assert lr_high < lr_low * 0.6


class TestFailureStreakBoost:
    """Test consecutive failure boost."""
    
    def test_few_failures_no_boost(self) -> None:
        """Few failures (< threshold) should not trigger boost."""
        calc = AdaptiveLearningRate(base_lr=0.05, failure_threshold=5)
        
        ctx_no_failures = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.0,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        ctx_few_failures = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.0,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=3,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        
        lr_no = calc.compute_adaptive_lr(error=5.0, context=ctx_no_failures)
        lr_few = calc.compute_adaptive_lr(error=5.0, context=ctx_few_failures)
        
        # Both should be similar (no boost)
        assert abs(lr_no - lr_few) < 0.01
    
    def test_many_failures_boost(self) -> None:
        """Many failures (>= threshold) should trigger 2x boost."""
        calc = AdaptiveLearningRate(base_lr=0.05, failure_threshold=5, failure_boost=2.0)
        
        ctx_no_failures = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.0,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        ctx_many_failures = PlasticityContext(
            regime=RegimeType.STABLE,
            noise_ratio=0.0,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=10,
            error_magnitude=5.0,
            is_critical_zone=False,
        )
        
        lr_no = calc.compute_adaptive_lr(error=5.0, context=ctx_no_failures)
        lr_many = calc.compute_adaptive_lr(error=5.0, context=ctx_many_failures)
        
        # Many failures should boost lr significantly
        assert lr_many > lr_no * 1.5


class TestSafetyLimits:
    """Test safety limits enforcement."""
    
    def test_lr_never_below_min(self) -> None:
        """Learning rate should never go below lr_min."""
        calc = AdaptiveLearningRate(base_lr=0.05, lr_min=0.001)
        
        # Extreme conditions that would normally produce very low lr
        ctx = PlasticityContext(
            regime=RegimeType.NOISY,
            noise_ratio=0.9,
            volatility=0.0,
            time_of_day=12,
            consecutive_failures=0,
            error_magnitude=0.01,
            is_critical_zone=False,
        )
        
        lr = calc.compute_adaptive_lr(error=0.01, context=ctx)
        
        assert lr >= 0.001
    
    def test_lr_never_above_max(self) -> None:
        """Learning rate should never go above lr_max."""
        calc = AdaptiveLearningRate(base_lr=0.05, lr_max=0.2)
        
        # Extreme conditions that would normally produce very high lr
        ctx = PlasticityContext(
            regime=RegimeType.VOLATILE,
            noise_ratio=0.0,
            volatility=1.0,
            time_of_day=12,
            consecutive_failures=20,
            error_magnitude=1000.0,
            is_critical_zone=False,
        )
        
        lr = calc.compute_adaptive_lr(error=1000.0, context=ctx)
        
        assert lr <= 0.2
    
    def test_lr_always_in_bounds(self) -> None:
        """Learning rate should always be in [lr_min, lr_max]."""
        calc = AdaptiveLearningRate()
        
        # Test with various random contexts
        import random
        random.seed(42)
        
        for _ in range(100):
            ctx = PlasticityContext(
                regime=random.choice(list(RegimeType)),
                noise_ratio=random.uniform(0, 1),
                volatility=random.uniform(0, 1),
                time_of_day=random.randint(0, 23),
                consecutive_failures=random.randint(0, 20),
                error_magnitude=random.uniform(0, 100),
                is_critical_zone=random.choice([True, False]),
            )
            error = random.uniform(0.1, 100)
            
            lr = calc.compute_adaptive_lr(error, ctx)
            
            assert 0.001 <= lr <= 0.2


class TestBatchComputation:
    """Test batch learning rate computation."""
    
    def test_batch_computation(self) -> None:
        """Test computing learning rates for multiple errors."""
        calc = AdaptiveLearningRate()
        
        errors = [1.0, 5.0, 10.0]
        contexts = [
            PlasticityContext.create_default(RegimeType.STABLE),
            PlasticityContext.create_default(RegimeType.VOLATILE),
            PlasticityContext.create_default(RegimeType.NOISY),
        ]
        
        lrs = calc.compute_batch_lr(errors, contexts)
        
        assert len(lrs) == 3
        assert all(0.001 <= lr <= 0.2 for lr in lrs)
    
    def test_batch_length_mismatch_raises(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        calc = AdaptiveLearningRate()
        
        errors = [1.0, 5.0]
        contexts = [PlasticityContext.create_default()]
        
        with pytest.raises(ValueError, match="same length"):
            calc.compute_batch_lr(errors, contexts)
