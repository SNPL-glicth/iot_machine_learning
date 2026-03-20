"""Tests for Maximum Likelihood Estimator."""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.inference.mle import (
    MaximumLikelihoodEstimator,
    MLEResult,
    fit_distribution,
)


class TestGaussianMLE:
    """Test Gaussian MLE."""
    
    def test_gaussian_fit_basic(self):
        """Test basic Gaussian fit."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="gaussian")
        
        assert result.distribution == "gaussian"
        assert abs(result.get_param("mu") - 3.0) < 0.01
        assert result.get_param("sigma2") > 0
        assert result.n_samples == 5
    
    def test_gaussian_single_point(self):
        """Test Gaussian with single point."""
        data = np.array([5.0])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="gaussian")
        
        assert result.get_param("mu") == 5.0
        assert result.get_param("sigma2") > 0  # Floor applied
    
    def test_gaussian_empty_data(self):
        """Test Gaussian with empty data."""
        data = np.array([])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="gaussian")
        
        assert result.n_samples == 0
        assert result.log_likelihood == -np.inf
    
    def test_gaussian_constant_data(self):
        """Test Gaussian with constant data (zero variance)."""
        data = np.array([3.0, 3.0, 3.0, 3.0])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="gaussian")
        
        assert result.get_param("mu") == 3.0
        assert result.get_param("sigma2") > 0  # Floored to avoid zero


class TestPoissonMLE:
    """Test Poisson MLE."""
    
    def test_poisson_fit_basic(self):
        """Test basic Poisson fit."""
        data = np.array([2, 3, 1, 4, 3, 2, 3])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="poisson")
        
        assert result.distribution == "poisson"
        expected_lambda = np.mean(data)
        assert abs(result.get_param("lambda") - expected_lambda) < 0.01
    
    def test_poisson_empty_data(self):
        """Test Poisson with empty data."""
        data = np.array([])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="poisson")
        
        assert result.n_samples == 0
        assert result.get_param("lambda") > 0


class TestBetaMLE:
    """Test Beta MLE."""
    
    def test_beta_fit_basic(self):
        """Test basic Beta fit."""
        data = np.array([0.2, 0.3, 0.25, 0.35, 0.28])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="beta")
        
        assert result.distribution == "beta"
        assert result.get_param("alpha") > 0
        assert result.get_param("beta") > 0
    
    def test_beta_edge_values(self):
        """Test Beta with values near 0 and 1."""
        data = np.array([0.01, 0.99, 0.5])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="beta")
        
        assert result.get_param("alpha") > 0
        assert result.get_param("beta") > 0


class TestExponentialMLE:
    """Test Exponential MLE."""
    
    def test_exponential_fit_basic(self):
        """Test basic Exponential fit."""
        data = np.array([1.0, 2.0, 1.5, 0.8, 1.2])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="exponential")
        
        assert result.distribution == "exponential"
        expected_lambda = 1.0 / np.mean(data)
        assert abs(result.get_param("lambda") - expected_lambda) < 0.01


class TestMLEEstimator:
    """Test unified MLE estimator."""
    
    def test_unsupported_distribution(self):
        """Test error on unsupported distribution."""
        data = np.array([1, 2, 3])
        
        mle = MaximumLikelihoodEstimator()
        
        with pytest.raises(ValueError, match="Unsupported distribution"):
            mle.fit(data, distribution="unknown")
    
    def test_fit_best_gaussian(self):
        """Test fit_best returns valid result for Gaussian-like data."""
        np.random.seed(42)
        data = np.random.normal(5.0, 2.0, 100)
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit_best(data)
        
        # Should return a valid distribution (any is acceptable)
        assert result.distribution in ["gaussian", "beta", "poisson", "exponential"]
        assert result.log_likelihood is not None
    
    def test_fit_best_with_candidates(self):
        """Test fit_best with subset of candidates."""
        data = np.array([1.0, 2.0, 3.0])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit_best(data, candidates=["gaussian", "exponential"])
        
        assert result.distribution in ["gaussian", "exponential"]
    
    def test_mle_result_interface(self):
        """Test MLEResult interface."""
        data = np.array([1.0, 2.0, 3.0])
        
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="gaussian")
        
        # Test get_param with default
        mu = result.get_param("mu", default=0.0)
        assert mu > 0
        
        # Test non-existent param
        missing = result.get_param("missing", default=99.0)
        assert missing == 99.0


class TestParameterFitter:
    """Test convenience wrapper."""
    
    def test_fit_distribution_wrapper(self):
        """Test fit_distribution returns params dict."""
        data = np.array([1.0, 2.0, 3.0])
        
        params = fit_distribution(data, distribution="gaussian")
        
        assert isinstance(params, dict)
        assert "mu" in params
        assert "sigma2" in params
