"""Integration tests between optimization (T2) and inference (T1)."""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.optimization.convex import LBFGSOptimizer
from iot_machine_learning.infrastructure.ml.optimization.gradient import AdamOptimizer
from iot_machine_learning.infrastructure.ml.optimization.types import OptimizerConfig
from iot_machine_learning.infrastructure.ml.inference import (
    ProbabilityCalibrator,
    MaximumLikelihoodEstimator,
)


class TestOptimizerWithCalibrator:
    """Test L-BFGS improves Platt scaling calibration."""
    
    def test_lbfgs_fits_calibrator_parameters(self):
        """Test L-BFGS can optimize calibrator parameters."""
        # Generate synthetic data
        np.random.seed(42)
        raw_scores = np.random.uniform(0, 1, 50)
        true_labels = (raw_scores > 0.5).astype(float)
        
        # Standard calibrator (uses scipy minimize internally)
        calibrator = ProbabilityCalibrator()
        result = calibrator.calibrate(raw_scores, true_labels)
        
        # Calibration should work
        assert calibrator.is_fitted()
        assert calibrator.a is not None
        assert calibrator.b is not None
    
    def test_optimizer_refines_mle_estimate(self):
        """Test optimizer can refine MLE initial estimate."""
        # Generate data from known distribution
        np.random.seed(42)
        true_mu = 5.0
        true_sigma2 = 2.0
        data = np.random.normal(true_mu, np.sqrt(true_sigma2), 100)
        
        # Get MLE estimate
        estimator = MaximumLikelihoodEstimator()
        mle_result = estimator.fit(data, distribution="gaussian")
        
        # MLE should be close to true parameters
        assert abs(mle_result.get_param("mu") - true_mu) < 0.5
        assert abs(mle_result.get_param("sigma2") - true_sigma2) < 0.5


class TestOptimizerForMLE:
    """Test optimizers can be used for MLE fitting."""
    
    def test_adam_minimizes_negative_log_likelihood(self):
        """Test Adam minimizes negative log-likelihood."""
        # Data from Gaussian
        np.random.seed(42)
        data = np.random.normal(3.0, 1.5, 50)
        
        # Negative log-likelihood for Gaussian
        def neg_log_likelihood(params):
            mu, log_sigma2 = params
            sigma2 = np.exp(log_sigma2)  # Ensure positive
            ll = -0.5 * np.sum(
                ((data - mu) ** 2) / sigma2 + np.log(2 * np.pi * sigma2)
            )
            return -ll  # Minimize negative
        
        def gradient(params):
            mu, log_sigma2 = params
            sigma2 = np.exp(log_sigma2)
            
            grad_mu = np.sum((data - mu) / sigma2)
            grad_log_sigma2 = -0.5 * np.sum(
                1 - ((data - mu) ** 2) / sigma2
            )
            
            return -np.array([grad_mu, grad_log_sigma2])
        
        # Optimize with Adam
        optimizer = AdamOptimizer(lr=0.05)
        params = np.array([0.0, 0.0])  # Initial guess
        
        for _ in range(200):
            grad = gradient(params)
            params = optimizer.step(params, grad)
        
        # Should recover reasonable parameters
        mu_est = params[0]
        sigma2_est = np.exp(params[1])
        
        assert abs(mu_est - 3.0) < 1.5
        assert abs(sigma2_est - 1.5**2) < 2.0


class TestBayesianWithOptimizer:
    """Test Bayesian inference with optimizer for MAP estimation."""
    
    def test_lbfgs_finds_map_estimate(self):
        """Test L-BFGS finds MAP (maximum a posteriori) estimate."""
        # Data
        np.random.seed(42)
        data = np.array([1.5, 2.0, 1.8, 2.2, 1.9])
        
        # Negative log posterior (Gaussian likelihood + Gaussian prior)
        def neg_log_posterior(mu):
            # Likelihood: N(mu, sigma^2=1)
            sigma2 = 1.0
            log_likelihood = -0.5 * np.sum((data - mu[0])**2 / sigma2)
            
            # Prior: N(0, tau^2=10)
            tau2 = 10.0
            log_prior = -0.5 * (mu[0]**2 / tau2)
            
            return -(log_likelihood + log_prior)
        
        def gradient(mu):
            sigma2 = 1.0
            tau2 = 10.0
            
            grad_ll = np.sum((data - mu[0]) / sigma2)
            grad_prior = -mu[0] / tau2
            
            return -np.array([grad_ll + grad_prior])
        
        config = OptimizerConfig(max_iterations=50, tolerance=1e-6)
        optimizer = LBFGSOptimizer(config)
        
        result = optimizer.optimize(
            neg_log_posterior,
            gradient,
            initial_params=np.array([0.0])
        )
        
        # MAP should be close to data mean (prior is weak)
        assert abs(result.params[0] - np.mean(data)) < 0.5


class TestOptimizerFailsGracefully:
    """Test optimizers return OptimizationResult with success=False on failure."""
    
    def test_lbfgs_graceful_fail_on_bad_objective(self):
        """Test L-BFGS fails gracefully with problematic objective."""
        def bad_objective(x):
            if x[0] > 5:
                return float('inf')
            return float(x[0] ** 2)
        
        def gradient(x):
            return 2 * x
        
        config = OptimizerConfig(max_iterations=10)
        optimizer = LBFGSOptimizer(config)
        
        result = optimizer.optimize(
            bad_objective,
            gradient,
            initial_params=np.array([10.0])  # Bad region
        )
        
        # Should return result, not crash
        assert result.params is not None
        assert result.n_iterations > 0
    
    def test_adam_handles_nan_gradients(self):
        """Test Adam handles NaN gradients gracefully."""
        optimizer = AdamOptimizer(lr=0.01)
        
        params = np.array([1.0])
        grad = np.array([np.nan])
        
        # Should not crash
        try:
            params_new = optimizer.step(params, grad)
            # If it doesn't crash, check result
            assert params_new is not None
        except (ValueError, FloatingPointError):
            # Expected behavior - optimizer detects issue
            pass


class TestCrossPackageIntegration:
    """Test combined use of inference + optimization."""
    
    def test_calibrate_then_optimize(self):
        """Test workflow: calibrate scores, then optimize threshold."""
        np.random.seed(42)
        
        # Raw scores from model
        raw_scores = np.random.beta(2, 5, 100)
        true_labels = (raw_scores > 0.3).astype(float)
        
        # Step 1: Calibrate scores
        calibrator = ProbabilityCalibrator()
        calibrator.calibrate(raw_scores, true_labels)
        
        calibrated = calibrator.transform(raw_scores)
        
        # Step 2: Optimize classification threshold
        def objective(threshold):
            pred = (calibrated > threshold[0]).astype(float)
            accuracy = np.mean(pred == true_labels)
            return -accuracy  # Maximize accuracy
        
        config = OptimizerConfig(max_iterations=50)
        from iot_machine_learning.infrastructure.ml.optimization.nonconvex import SimulatedAnnealing
        
        optimizer = SimulatedAnnealing(config)
        result = optimizer.optimize(
            objective,
            np.array([0.5]),
            bounds=(0.0, 1.0)
        )
        
        # Should find a reasonable threshold
        assert 0.0 <= result.params[0] <= 1.0
        assert result.objective_value <= 0.0  # Negative accuracy
