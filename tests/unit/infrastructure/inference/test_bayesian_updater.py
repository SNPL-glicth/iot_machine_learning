"""Tests for Bayesian updater."""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.inference.bayesian import (
    BayesianUpdater,
    GaussianPrior,
    BetaPrior,
    GammaPrior,
    Posterior,
)


class TestGaussianUpdate:
    """Test Gaussian conjugate update."""
    
    def test_gaussian_update_basic(self):
        """Test basic Gaussian update."""
        updater = BayesianUpdater()
        prior = GaussianPrior(mu_0=0.0, sigma2_0=1.0)
        observations = np.array([1.0, 2.0, 3.0])
        
        posterior = updater.update(prior, observations)
        
        assert posterior.distribution == "gaussian"
        assert posterior.n_observations == 3
        
        # Posterior mean should be between prior mean and data mean
        mu_post = posterior.get_param("mu_0")
        assert 0.0 < mu_post < 2.0
    
    def test_gaussian_update_empty(self):
        """Test Gaussian update with no observations."""
        updater = BayesianUpdater()
        prior = GaussianPrior(mu_0=5.0, sigma2_0=2.0)
        observations = np.array([])
        
        posterior = updater.update(prior, observations)
        
        # Posterior should equal prior
        assert posterior.get_param("mu_0") == 5.0
        assert posterior.get_param("sigma2_0") == 2.0
    
    def test_gaussian_sequential_updates(self):
        """Test sequential Bayesian updates."""
        updater = BayesianUpdater()
        prior = GaussianPrior(mu_0=0.0, sigma2_0=10.0)
        
        # First update
        obs1 = np.array([1.0, 2.0])
        posterior1 = updater.update(prior, obs1)
        
        # Second update (posterior becomes prior)
        obs2 = np.array([3.0, 4.0])
        prior2 = posterior1.to_prior()
        posterior2 = updater.update(prior2, obs2)
        
        # Variance should decrease with more data
        assert posterior2.get_param("sigma2_0") < posterior1.get_param("sigma2_0")
    
    def test_gaussian_convergence(self):
        """Test posterior converges to data with many observations."""
        updater = BayesianUpdater()
        prior = GaussianPrior(mu_0=0.0, sigma2_0=1.0)
        
        # Many observations
        np.random.seed(42)
        observations = np.random.normal(5.0, 0.1, 100)
        
        posterior = updater.update(prior, observations)
        
        # Posterior mean should be close to data mean
        mu_post = posterior.get_param("mu_0")
        assert abs(mu_post - 5.0) < 0.5


class TestBetaUpdate:
    """Test Beta conjugate update."""
    
    def test_beta_update_basic(self):
        """Test basic Beta update."""
        updater = BayesianUpdater()
        prior = BetaPrior(alpha=1.0, beta=1.0)  # Uniform prior
        observations = np.array([0.8, 0.9, 0.7, 0.85])
        
        posterior = updater.update(prior, observations)
        
        assert posterior.distribution == "beta"
        assert posterior.n_observations == 4
        
        # Alpha should increase more than beta (high values)
        alpha = posterior.get_param("alpha")
        beta_param = posterior.get_param("beta")
        assert alpha > beta_param
    
    def test_beta_update_low_values(self):
        """Test Beta update with low values."""
        updater = BayesianUpdater()
        prior = BetaPrior(alpha=1.0, beta=1.0)
        observations = np.array([0.1, 0.2, 0.15])
        
        posterior = updater.update(prior, observations)
        
        # Beta should increase more than alpha (low values)
        alpha = posterior.get_param("alpha")
        beta_param = posterior.get_param("beta")
        assert beta_param > alpha


class TestGammaUpdate:
    """Test Gamma conjugate update."""
    
    def test_gamma_update_basic(self):
        """Test basic Gamma update."""
        updater = BayesianUpdater()
        prior = GammaPrior(alpha=1.0, beta=1.0)
        observations = np.array([2, 3, 2, 4, 3])
        
        posterior = updater.update(prior, observations)
        
        assert posterior.distribution == "gamma"
        assert posterior.n_observations == 5
        
        # Alpha increases by sum of observations
        alpha = posterior.get_param("alpha")
        assert alpha > 1.0
        
        # Beta increases by number of observations
        beta_param = posterior.get_param("beta")
        assert beta_param == 6.0  # 1.0 + 5


class TestPosteriorPredictive:
    """Test posterior predictive distributions."""
    
    def test_gaussian_predictive(self):
        """Test Gaussian posterior predictive."""
        updater = BayesianUpdater()
        prior = GaussianPrior(mu_0=0.0, sigma2_0=1.0)
        observations = np.array([1.0, 2.0, 3.0])
        
        posterior = updater.update(prior, observations)
        
        # Predict probability of new observation
        prob = updater.predict_probability(posterior, 2.5)
        
        assert 0.0 <= prob <= 1.0
    
    def test_beta_predictive(self):
        """Test Beta posterior predictive."""
        updater = BayesianUpdater()
        prior = BetaPrior(alpha=2.0, beta=2.0)
        observations = np.array([0.6, 0.7, 0.65])
        
        posterior = updater.update(prior, observations)
        
        # Predict probability of new observation
        prob = updater.predict_probability(posterior, 0.68)
        
        assert 0.0 <= prob <= 1.0


class TestPosteriorToPrior:
    """Test posterior to prior conversion."""
    
    def test_posterior_to_prior_conversion(self):
        """Test converting posterior to prior."""
        updater = BayesianUpdater()
        prior = GaussianPrior(mu_0=0.0, sigma2_0=1.0)
        observations = np.array([1.0, 2.0])
        
        posterior = updater.update(prior, observations)
        new_prior = posterior.to_prior()
        
        assert new_prior.distribution == posterior.distribution
        assert new_prior.parameters == posterior.parameters


class TestUnsupportedDistribution:
    """Test error handling."""
    
    def test_unsupported_prior(self):
        """Test error on unsupported prior distribution."""
        updater = BayesianUpdater()
        
        # Create invalid prior
        from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import Prior
        invalid_prior = Prior(distribution="unknown", parameters={})
        
        observations = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Unsupported prior"):
            updater.update(invalid_prior, observations)
