"""Tests for per-engine sigma2_obs estimation in BayesianWeightTracker.

Validates:
- Empirical variance used after min_samples
- Fallback to default with insufficient samples
- Minimum floor enforced
- Different sigma2_obs per engine
"""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker import (
    BayesianWeightTracker,
)


class TestSigma2ObsEstimation:
    """Test sigma2_obs per-engine estimation."""

    def test_sigma2_obs_uses_empirical_variance_after_min_samples(self) -> None:
        """After 5+ errors, sigma2_obs should be empirical variance."""
        tracker = BayesianWeightTracker(
            variance_window=20,
            variance_min_samples=5,
            sigma2_obs_default=1.0,
            sigma2_obs_min=0.01,
        )
        # Record 5 identical errors → variance = 0 → clamped to min 0.01
        for _ in range(5):
            tracker.update("STABLE", "engine_a", 2.0)

        sigma2 = tracker._variance_estimator.get_sigma2_obs("engine_a")
        assert sigma2 == pytest.approx(0.01, rel=1e-6)
        # source check via internal deque length
        assert len(tracker._variance_estimator._deques["engine_a"]) == 5

    def test_sigma2_obs_falls_back_to_default_with_insufficient_samples(self) -> None:
        """With <5 errors, sigma2_obs should fall back to default."""
        tracker = BayesianWeightTracker(
            variance_window=20,
            variance_min_samples=5,
            sigma2_obs_default=1.0,
            sigma2_obs_min=0.01,
        )
        tracker.update("STABLE", "engine_b", 1.0)
        tracker.update("STABLE", "engine_b", 2.0)
        tracker.update("STABLE", "engine_b", 3.0)

        sigma2 = tracker._variance_estimator.get_sigma2_obs("engine_b")
        assert sigma2 == pytest.approx(1.0, rel=1e-6)
        assert len(tracker._variance_estimator._deques["engine_b"]) == 3

    def test_sigma2_obs_never_below_minimum(self) -> None:
        """Empirical variance must be clamped to sigma2_obs_min."""
        tracker = BayesianWeightTracker(
            variance_window=20,
            variance_min_samples=5,
            sigma2_obs_default=1.0,
            sigma2_obs_min=0.05,
        )
        # 10 identical errors → raw variance = 0
        for _ in range(10):
            tracker.update("STABLE", "engine_c", 5.0)

        sigma2 = tracker._variance_estimator.get_sigma2_obs("engine_c")
        assert sigma2 >= 0.05
        assert sigma2 == pytest.approx(0.05, rel=1e-6)

    def test_sigma2_obs_different_per_engine(self) -> None:
        """Two engines with different error spreads produce different sigma2."""
        tracker = BayesianWeightTracker(
            variance_window=20,
            variance_min_samples=5,
            sigma2_obs_default=1.0,
            sigma2_obs_min=0.01,
        )
        # Engine d: errors 1,2,3,4,5 → variance = 2.0
        for err in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.update("STABLE", "engine_d", err)
        # Engine e: errors 10,10,10,10,10 → variance = 0 → clamped 0.01
        for _ in range(5):
            tracker.update("STABLE", "engine_e", 10.0)

        sigma2_d = tracker._variance_estimator.get_sigma2_obs("engine_d")
        sigma2_e = tracker._variance_estimator.get_sigma2_obs("engine_e")

        assert sigma2_d > sigma2_e
        assert sigma2_d == pytest.approx(2.0, rel=1e-6)
        assert sigma2_e == pytest.approx(0.01, rel=1e-6)
