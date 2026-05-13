"""Tests for core/robust_statistics.py."""

import numpy as np
import pytest

from core.statistical.robust_statistics import RobustStatistics


class TestRobustStatistics:
    def test_median_absolute_deviation_normal(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        mad = RobustStatistics.median_absolute_deviation(data)
        # MAD ≈ 0.6745 * σ under normality (allow some variance)
        expected_mad = 0.6745 * np.std(data)
        assert abs(mad - expected_mad) < 0.2

    def test_median_absolute_deviation_skewed(self):
        np.random.seed(42)
        data = np.random.exponential(1, 100)
        mad = RobustStatistics.median_absolute_deviation(data)
        assert mad > 0
        assert mad < np.std(data)  # MAD should be smaller than std for skewed

    def test_robust_z_score_normal_value(self):
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        value = 12.0
        z = RobustStatistics.robust_z_score(data, value)
        # Should be approximately 1σ from median
        assert abs(z) < 3.0

    def test_robust_z_score_outlier(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        value = 10.0
        z = RobustStatistics.robust_z_score(data, value)
        # Should be large for outlier
        assert z > 5.0

    def test_robust_z_score_division_by_zero(self):
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        value = 5.0
        z = RobustStatistics.robust_z_score(data, value)
        # Should return 0 when MAD is 0
        assert z == 0.0

    def test_robust_outlier_bounds_normal(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        lower, upper = RobustStatistics.robust_outlier_bounds(data, k=3.0)
        # Bounds should be approximately ±3σ
        median = np.median(data)
        assert lower < median
        assert upper > median
        assert upper - lower > 4.0

    def test_robust_outlier_bounds_custom_k(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        lower_2, upper_2 = RobustStatistics.robust_outlier_bounds(data, k=2.0)
        lower_3, upper_3 = RobustStatistics.robust_outlier_bounds(data, k=3.0)
        # k=2 should give tighter bounds than k=3
        assert (upper_2 - lower_2) < (upper_3 - lower_3)

    def test_robust_outlier_bounds_skewed(self):
        np.random.seed(42)
        data = np.random.exponential(1, 100)
        lower, upper = RobustStatistics.robust_outlier_bounds(data, k=3.0)
        median = np.median(data)
        # For skewed data, bounds should still be valid
        assert lower < median
        assert upper > median
        assert upper > 0
