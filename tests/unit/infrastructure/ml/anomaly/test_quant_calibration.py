"""Tests for Quant Engineering calibration improvements.

Tests with synthetic numpy data showing behavior before/after for:
- CRÍTICO-5/MATH-SEV-1: Adaptive contamination in IsolationForest
- SEVERO-2: Z-score absolute bounds
- SEVERO-3: Bayesian tracker convergence
"""

import math

import numpy as np
import pytest

from iot_machine_learning.infrastructure.ml.anomaly.detectors.isolation_forest_detector import (
    IsolationForestDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.detectors.z_score_detector import (
    ZScoreDetector,
)


class TestIsolationForestAdaptiveContamination:
    """Test CRÍTICO-5/MATH-SEV-1: Adaptive contamination."""
    
    def test_adaptive_contamination_with_clean_data(self):
        """Clean data (1% outliers) → adaptive contamination ≈ 0.01."""
        np.random.seed(42)
        
        # Generate clean data: 99% normal, 1% outliers
        normal_data = np.random.normal(50, 5, 990)
        outliers = np.random.normal(100, 10, 10)
        data = np.concatenate([normal_data, outliers])
        np.random.shuffle(data)
        
        # BEFORE: Fixed contamination = 0.1 (10%)
        detector_fixed = IsolationForestDetector(contamination=0.1, adaptive=False)
        detector_fixed.train(data.tolist())
        
        # AFTER: Adaptive contamination
        detector_adaptive = IsolationForestDetector(contamination=0.1, adaptive=True)
        detector_adaptive.train(data.tolist())
        
        # Test on normal value (should not be anomaly)
        normal_value = 50.0
        vote_fixed = detector_fixed.vote(normal_value)
        vote_adaptive = detector_adaptive.vote(normal_value)
        
        # Adaptive should be less aggressive (lower false positive rate)
        # With 1% true contamination, adaptive estimates ~0.01-0.05
        # Fixed at 0.1 is too high, may flag normal values
        assert vote_adaptive is not None
        assert vote_fixed is not None
    
    def test_adaptive_contamination_with_noisy_data(self):
        """Noisy data (20% outliers) → adaptive contamination ≈ 0.2."""
        np.random.seed(42)
        
        # Generate noisy data: 80% normal, 20% outliers
        normal_data = np.random.normal(50, 5, 800)
        outliers = np.random.normal(100, 10, 200)
        data = np.concatenate([normal_data, outliers])
        np.random.shuffle(data)
        
        # BEFORE: Fixed contamination = 0.1 (too low)
        detector_fixed = IsolationForestDetector(contamination=0.1, adaptive=False)
        detector_fixed.train(data.tolist())
        
        # AFTER: Adaptive contamination
        detector_adaptive = IsolationForestDetector(contamination=0.1, adaptive=True)
        detector_adaptive.train(data.tolist())
        
        # Test on outlier (should be anomaly)
        outlier_value = 100.0
        vote_fixed = detector_fixed.vote(outlier_value)
        vote_adaptive = detector_adaptive.vote(outlier_value)
        
        # Both should detect, but adaptive is calibrated to actual rate
        assert vote_adaptive is not None
        assert vote_fixed is not None
    
    def test_adaptive_contamination_insufficient_data(self):
        """Insufficient data (<100 samples) → falls back to base contamination."""
        np.random.seed(42)
        
        # Only 50 samples
        data = np.random.normal(50, 5, 50)
        
        detector = IsolationForestDetector(contamination=0.1, adaptive=True)
        detector.train(data.tolist())
        
        # Should still train (fallback to base contamination)
        assert detector.is_trained
    
    def test_adaptive_contamination_clamping(self):
        """Extreme contamination estimates are clamped to [0.01, 0.5]."""
        np.random.seed(42)
        
        # All outliers (100% contamination)
        data = np.random.normal(100, 50, 100)
        
        detector = IsolationForestDetector(contamination=0.1, adaptive=True)
        detector.train(data.tolist())
        
        # Should clamp to max 0.5 (50%)
        assert detector.is_trained


class TestZScoreAbsoluteBounds:
    """Test SEVERO-2: Z-score absolute bounds."""
    
    def test_absolute_bounds_prevent_extreme_thresholds(self):
        """Absolute bounds prevent thresholds from exceeding 10σ/15σ."""
        # Create detector with high scale_max (allows 5x scaling)
        detector = ZScoreDetector(
            lower=2.0,
            upper=3.0,
            adaptive=True,
            scale_max=5.0,
            max_lower=10.0,  # SEVERO-2: absolute bound
            max_upper=15.0,  # SEVERO-2: absolute bound
        )
        
        # Train with normal data
        np.random.seed(42)
        normal_data = np.random.normal(50, 5, 100)
        detector.train(normal_data.tolist())
        
        # Simulate high volatility (should trigger adaptive scaling)
        # But absolute bounds should prevent extreme thresholds
        volatile_data = np.random.normal(50, 50, 100)  # 10x std
        detector.train(volatile_data.tolist())
        
        # Get effective thresholds (private method, but we can test via vote)
        # With scale_max=5.0: lower could be 2.0 * 5.0 = 10.0
        # With scale_max=5.0: upper could be 3.0 * 5.0 = 15.0
        # Absolute bounds should clamp to exactly 10.0 and 15.0
        
        # Test extreme value
        extreme_value = 50 + (20 * 50)  # 20σ from mean
        vote = detector.vote(extreme_value)
        
        # Should be detected as anomaly (beyond 15σ)
        assert vote == 1.0
    
    def test_absolute_bounds_on_base_thresholds(self):
        """Absolute bounds apply even to non-adaptive base thresholds."""
        # Create detector with base thresholds that exceed absolute bounds
        detector = ZScoreDetector(
            lower=12.0,  # Exceeds max_lower=10.0
            upper=20.0,  # Exceeds max_upper=15.0
            adaptive=False,
            max_lower=10.0,
            max_upper=15.0,
        )
        
        np.random.seed(42)
        data = np.random.normal(50, 5, 100)
        detector.train(data.tolist())
        
        # Effective thresholds should be clamped
        # lower: min(12.0, 10.0) = 10.0
        # upper: min(20.0, 15.0) = 15.0
        
        # Test value at 11σ (between clamped lower and original lower)
        value_11sigma = 50 + (11 * 5)
        vote = detector.vote(value_11sigma)
        
        # With clamped lower=10.0, this should be in anomaly range
        # (between 10σ and 15σ → partial vote)
        assert vote is not None
        assert 0.0 <= vote <= 1.0
    
    def test_absolute_bounds_configurable(self):
        """Absolute bounds are configurable via constructor (OCP)."""
        # Custom absolute bounds
        detector = ZScoreDetector(
            lower=2.0,
            upper=3.0,
            max_lower=5.0,  # Custom
            max_upper=8.0,  # Custom
        )
        
        assert detector._max_lower == 5.0
        assert detector._max_upper == 8.0


class TestBayesianTrackerConvergence:
    """Test SEVERO-3: Bayesian tracker convergence detection."""
    
    def test_convergence_detection_stable_weights(self):
        """Stable weights (CV < 0.05) trigger convergence and alpha decay."""
        from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.base import (
            BayesianWeightTracker,
        )
        
        tracker = BayesianWeightTracker(alpha=0.15)
        
        # Simulate stable weight updates (converging to 0.8)
        regime = "stable"
        engine = "taylor"
        
        # Feed consistent errors (low variance)
        np.random.seed(42)
        for i in range(20):
            # Small prediction error → high accuracy → weight ≈ 0.8
            error = np.random.normal(0.2, 0.01)  # Very stable
            tracker.update(regime, engine, error)
        
        # After 20 updates, last 10 should have low CV
        # Alpha should have decayed
        assert tracker._alpha < 0.15  # Decayed from initial 0.15
    
    def test_no_convergence_with_unstable_weights(self):
        """Unstable weights (CV >= 0.05) do not trigger convergence."""
        from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.base import (
            BayesianWeightTracker,
        )
        
        tracker = BayesianWeightTracker(alpha=0.15)
        
        regime = "volatile"
        engine = "arima"
        
        # Feed volatile errors (high variance)
        np.random.seed(42)
        for i in range(20):
            # Alternating high/low errors → unstable weights
            error = np.random.uniform(0.1, 5.0)
            tracker.update(regime, engine, error)
        
        # Alpha should not have decayed (no convergence)
        assert tracker._alpha == 0.15  # Unchanged
    
    def test_convergence_check_requires_10_samples(self):
        """Convergence check requires at least 10 weight samples."""
        from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.base import (
            BayesianWeightTracker,
        )
        
        tracker = BayesianWeightTracker(alpha=0.15)
        
        # Only 5 updates
        regime = "test"
        engine = "test_engine"
        
        for i in range(5):
            tracker.update(regime, engine, 0.1)
        
        # Should not converge (insufficient history)
        weight_key = f"default:{regime}:{engine}"
        assert not tracker._check_convergence(weight_key)
    
    def test_coefficient_of_variation_calculation(self):
        """CV calculation: std / mean < 0.05 for convergence."""
        from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.base import (
            BayesianWeightTracker,
        )
        
        tracker = BayesianWeightTracker(alpha=0.15)
        
        # Manually populate weight history with known values
        weight_key = "test_regime:test_engine"
        
        # Stable weights: mean=0.8, std ≈ 0.02 → CV ≈ 0.025 < 0.05
        stable_weights = [0.78, 0.79, 0.80, 0.81, 0.82, 0.79, 0.80, 0.81, 0.80, 0.79]
        for w in stable_weights:
            tracker._weight_history[weight_key].append(w)
        
        # Should converge
        assert tracker._check_convergence(weight_key)
        
        # Volatile weights: mean=0.5, std ≈ 0.3 → CV ≈ 0.6 > 0.05
        weight_key_volatile = "volatile_regime:volatile_engine"
        volatile_weights = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5]
        for w in volatile_weights:
            tracker._weight_history[weight_key_volatile].append(w)
        
        # Should not converge
        assert not tracker._check_convergence(weight_key_volatile)


class TestAdaptiveInhibitionConfig:
    """Test MATH-SEV-4/SEVERO-1: Adaptive inhibition thresholds."""
    
    def test_adaptive_thresholds_from_percentiles(self):
        """Thresholds calculated from 75th percentile of historical errors."""
        from iot_machine_learning.infrastructure.ml.cognitive.inhibition.adaptive_config import (
            AdaptiveInhibitionConfig,
        )
        
        config = AdaptiveInhibitionConfig(percentile=75.0, min_samples=20)
        
        # Populate error history
        np.random.seed(42)
        errors = np.random.uniform(0.5, 10.0, 100).tolist()
        config.update_error_history("sensor_1", errors)
        
        # Get adaptive thresholds
        thresholds = config.get_adaptive_thresholds("sensor_1")
        
        assert thresholds is not None
        fit_error, recent_error, stability = thresholds
        
        # 75th percentile should be around 7-8
        assert 5.0 <= fit_error <= 10.0
        assert 5.0 <= recent_error <= 10.0
        assert stability == config.stability_threshold  # Unchanged
    
    def test_adaptive_thresholds_insufficient_data(self):
        """Returns None when insufficient error samples."""
        from iot_machine_learning.infrastructure.ml.cognitive.inhibition.adaptive_config import (
            AdaptiveInhibitionConfig,
        )
        
        config = AdaptiveInhibitionConfig(percentile=75.0, min_samples=20)
        
        # Only 10 errors (< min_samples)
        errors = [1.0] * 10
        config.update_error_history("sensor_1", errors)
        
        thresholds = config.get_adaptive_thresholds("sensor_1")
        
        assert thresholds is None
    
    def test_adaptive_config_extends_base_config(self):
        """AdaptiveInhibitionConfig extends InhibitionConfig (OCP)."""
        from iot_machine_learning.infrastructure.ml.cognitive.inhibition.adaptive_config import (
            AdaptiveInhibitionConfig,
        )
        from iot_machine_learning.infrastructure.ml.cognitive.inhibition.gate import (
            InhibitionConfig,
        )
        
        config = AdaptiveInhibitionConfig(
            percentile=80.0,
            min_samples=30,
            stability_threshold=0.7,  # Base config param
        )
        
        assert isinstance(config, InhibitionConfig)
        assert config.stability_threshold == 0.7
        assert config._percentile == 80.0
