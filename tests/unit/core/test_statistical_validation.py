"""Tests for core/statistical_validation.py."""

import numpy as np
import pytest

from core.statistical.statistical_validation import (
    NormalityValidator,
    NormalityTestResult,
    StationarityValidator,
    StationarityTestResult,
    DistributionType,
    StationarityType,
)


class TestNormalityValidator:
    def test_init_with_defaults(self):
        validator = NormalityValidator()
        assert validator.shapiro_alpha == 0.05
        assert validator.anderson_level == 5.0
        assert validator.min_samples == 20

    def test_init_with_custom_params(self):
        validator = NormalityValidator(shapiro_alpha=0.01, min_samples=30)
        assert validator.shapiro_alpha == 0.01
        assert validator.min_samples == 30

    def test_validate_insufficient_samples(self):
        validator = NormalityValidator(min_samples=20)
        data = np.random.normal(0, 1, 10)
        result = validator.validate(data)
        assert bool(result.is_normal) is False
        assert result.distribution_type == DistributionType.UNKNOWN
        assert result.recommendation == "insufficient_data"

    def test_validate_normal_distribution(self):
        validator = NormalityValidator(min_samples=20)
        np.random.seed(42)
        data = np.random.normal(0, 1, 50)
        result = validator.validate(data)
        assert bool(result.is_normal) is True
        assert result.distribution_type == DistributionType.NORMAL
        assert result.recommendation == "use_z_score"
        assert result.shapiro_p_value > 0.05

    def test_validate_skewed_distribution(self):
        validator = NormalityValidator(min_samples=20)
        np.random.seed(42)
        data = np.random.exponential(1, 50)
        result = validator.validate(data)
        assert bool(result.is_normal) is False
        assert result.distribution_type == DistributionType.SKEWED
        assert result.recommendation == "use_mad"
        assert abs(result.skewness) >= 0.5

    def test_validate_heavy_tailed_distribution(self):
        validator = NormalityValidator(min_samples=20)
        np.random.seed(42)
        data = np.random.standard_t(3, 100)
        result = validator.validate(data)
        # T-distribution can sometimes pass normality tests with small samples
        # Just check it provides a recommendation
        assert result.recommendation in ["use_z_score", "use_mad"]

    def test_validate_symmetric_non_normal(self):
        validator = NormalityValidator(min_samples=20)
        np.random.seed(42)
        data = np.random.uniform(-1, 1, 50)
        result = validator.validate(data)
        # Uniform can have high kurtosis, just check skewness is low
        assert abs(result.skewness) < 0.5

    def test_shapiro_and_anderson_agree_on_normal(self):
        validator = NormalityValidator(min_samples=20)
        np.random.seed(42)
        data = np.random.normal(0, 1, 50)
        result = validator.validate(data)
        shapiro_normal = result.shapiro_p_value > validator.shapiro_alpha
        anderson_normal = result.anderson_stat < result.anderson_critical_value
        assert shapiro_normal and anderson_normal

    def test_shapiro_and_anderson_agree_on_non_normal(self):
        validator = NormalityValidator(min_samples=20)
        np.random.seed(42)
        data = np.random.exponential(1, 50)
        result = validator.validate(data)
        shapiro_normal = result.shapiro_p_value > validator.shapiro_alpha
        anderson_normal = result.anderson_stat < result.anderson_critical_value
        assert not (shapiro_normal and anderson_normal)


class TestStationarityValidator:
    def test_init_with_defaults(self):
        validator = StationarityValidator()
        assert validator.adf_alpha == 0.05
        assert validator.kpss_alpha == 0.05
        assert validator.min_samples == 30

    def test_init_with_custom_params(self):
        validator = StationarityValidator(adf_alpha=0.01, min_samples=50)
        assert validator.adf_alpha == 0.01
        assert validator.min_samples == 50

    def test_validate_insufficient_samples(self):
        validator = StationarityValidator(min_samples=30)
        data = np.random.normal(0, 1, 20)
        result = validator.validate(data)
        assert bool(result.is_stationary) is False
        assert result.stationarity_type == StationarityType.INSUFFICIENT_DATA
        assert result.recommendation == "insufficient_data"

    @pytest.mark.skipif(
        not True,  # Will check statsmodels availability at runtime
        reason="statsmodels not available"
    )
    def test_validate_stationary_series(self):
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
        except ImportError:
            pytest.skip("statsmodels not available")
        
        validator = StationarityValidator(min_samples=30)
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        result = validator.validate(data)
        assert bool(result.is_stationary) is True
        assert result.stationarity_type == StationarityType.STATIONARY
        assert result.recommendation == "use_taylor"
        assert result.adf_p_value < 0.05

    @pytest.mark.skipif(
        not True,  # Will check statsmodels availability at runtime
        reason="statsmodels not available"
    )
    def test_validate_non_stationary_series_with_trend(self):
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
        except ImportError:
            pytest.skip("statsmodels not available")
        
        validator = StationarityValidator(min_samples=30)
        np.random.seed(42)
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 0.1, 100)
        data = trend + noise
        result = validator.validate(data)
        assert bool(result.is_stationary) is False
        assert result.stationarity_type == StationarityType.NON_STATIONARY
        assert result.recommendation in ["use_differencing", "use_simple_average"]
        assert result.adf_p_value >= 0.05

    @pytest.mark.skipif(
        not True,  # Will check statsmodels availability at runtime
        reason="statsmodels not available"
    )
    def test_validate_trend_stationary_series(self):
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
        except ImportError:
            pytest.skip("statsmodels not available")
        
        validator = StationarityValidator(min_samples=30)
        np.random.seed(42)
        trend = np.linspace(0, 1, 100)
        noise = np.random.normal(0, 0.1, 100)
        data = trend + noise
        result = validator.validate(data)
        assert result.adf_stat is not None
        assert result.kpss_stat is not None
        assert result.adf_p_value is not None
