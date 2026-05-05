"""Bayesian posterior computation via conjugate priors.

Conjugate pairs:
- Gaussian-Gaussian: Gaussian likelihood + Gaussian prior → Gaussian posterior
- Beta-Bernoulli: Bernoulli likelihood + Beta prior → Beta posterior
- Gamma-Poisson: Poisson likelihood + Gamma prior → Gamma posterior
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from .prior import Prior, GaussianPrior, BetaPrior, GammaPrior


@dataclass
class Posterior:
    """Posterior distribution after Bayesian update."""
    distribution: str
    parameters: Dict[str, float]
    n_observations: int
    
    def get_param(self, name: str, default: float = 0.0) -> float:
        """Get parameter by name."""
        return self.parameters.get(name, default)
    
    def to_prior(self) -> Prior:
        """Convert posterior to prior for next update."""
        return Prior(
            distribution=self.distribution,
            parameters=self.parameters.copy(),
        )


class BayesianUpdater:
    """Bayesian updater using conjugate priors.
    
    Implements closed-form posterior updates for conjugate pairs.
    
    Example:
        updater = BayesianUpdater()
        prior = GaussianPrior(mu_0=0.0, sigma2_0=1.0)
        observations = np.array([0.5, 0.7, 0.6])
        posterior = updater.update(prior, observations)
    """
    
    def update(
        self,
        prior: Prior,
        observations: np.ndarray,
        sigma2_obs: Optional[float] = None,
    ) -> Posterior:
        """Update prior with observations → posterior.

        Uses conjugate prior formulas for closed-form update.

        Args:
            prior: Prior distribution
            observations: New observations
            sigma2_obs: Known observation variance (default 1.0 for backward compat)

        Returns:
            Posterior distribution
        """
        if len(observations) == 0:
            # No observations → posterior = prior
            return Posterior(
                distribution=prior.distribution,
                parameters=prior.parameters.copy(),
                n_observations=0,
            )

        if prior.distribution == "gaussian":
            return self._update_gaussian(prior, observations, sigma2_obs)
        elif prior.distribution == "beta":
            return self._update_beta(prior, observations)
        elif prior.distribution == "gamma":
            return self._update_gamma(prior, observations)
        else:
            raise ValueError(f"Unsupported prior: {prior.distribution}")

    def _update_gaussian(
        self,
        prior: Prior,
        observations: np.ndarray,
        sigma2_obs: Optional[float] = None,
    ) -> Posterior:
        """Gaussian-Gaussian conjugate update.

        Prior: N(μ_0, σ²_0)
        Likelihood: N(μ, σ²_known)
        Posterior: N(μ_n, σ²_n)

        sigma2_obs is no longer hardcoded; callers should provide the per-engine
        empirical variance for more accurate updates.
        """
        mu_0 = prior.get_param("mu_0", 0.0)
        sigma2_0 = prior.get_param("sigma2_0", 1.0)

        n = len(observations)
        sum_x = np.sum(observations)

        effective_sigma2_obs = sigma2_obs if sigma2_obs is not None else 1.0

        # Posterior parameters
        precision_0 = 1.0 / sigma2_0
        precision_obs = n / effective_sigma2_obs
        precision_n = precision_0 + precision_obs

        mu_n = (precision_0 * mu_0 + precision_obs * (sum_x / n)) / precision_n
        sigma2_n = 1.0 / precision_n

        return Posterior(
            distribution="gaussian",
            parameters={"mu_0": mu_n, "sigma2_0": sigma2_n},
            n_observations=n,
        )
    
    def _update_beta(
        self,
        prior: Prior,
        observations: np.ndarray,
    ) -> Posterior:
        """Beta-Bernoulli conjugate update.
        
        Prior: Beta(α, β)
        Likelihood: Bernoulli(p)
        Posterior: Beta(α + successes, β + failures)
        
        For continuous [0, 1] data, we treat as weighted update.
        """
        alpha_0 = prior.get_param("alpha", 1.0)
        beta_0 = prior.get_param("beta", 1.0)
        
        # For continuous data in [0, 1], use pseudo-counts
        # Each observation x contributes x to alpha, (1-x) to beta
        sum_x = np.sum(observations)
        sum_1_minus_x = np.sum(1 - observations)
        
        alpha_n = alpha_0 + sum_x
        beta_n = beta_0 + sum_1_minus_x
        
        return Posterior(
            distribution="beta",
            parameters={"alpha": alpha_n, "beta": beta_n},
            n_observations=len(observations),
        )
    
    def _update_gamma(
        self,
        prior: Prior,
        observations: np.ndarray,
    ) -> Posterior:
        """Gamma-Poisson conjugate update.
        
        Prior: Gamma(α, β)
        Likelihood: Poisson(λ)
        Posterior: Gamma(α + Σx_i, β + n)
        """
        alpha_0 = prior.get_param("alpha", 1.0)
        beta_0 = prior.get_param("beta", 1.0)
        
        n = len(observations)
        sum_x = np.sum(observations)
        
        alpha_n = alpha_0 + sum_x
        beta_n = beta_0 + n
        
        return Posterior(
            distribution="gamma",
            parameters={"alpha": alpha_n, "beta": beta_n},
            n_observations=n,
        )
    
    def predict_probability(
        self,
        posterior: Posterior,
        new_observation: float,
    ) -> float:
        """Posterior predictive: P(x_new | data).
        
        Args:
            posterior: Posterior after observing data
            new_observation: New observation to score
            
        Returns:
            Probability (or log-probability for continuous)
        """
        if posterior.distribution == "gaussian":
            mu = posterior.get_param("mu_0", 0.0)
            sigma2 = posterior.get_param("sigma2_0", 1.0)
            
            # Predictive is Student-t, approximate with Gaussian
            log_prob = -0.5 * (
                ((new_observation - mu) ** 2) / sigma2 +
                np.log(2 * np.pi * sigma2)
            )
            return float(np.exp(log_prob))
        
        elif posterior.distribution == "beta":
            alpha = posterior.get_param("alpha", 1.0)
            beta_param = posterior.get_param("beta", 1.0)
            
            # Predictive mean
            mean = alpha / (alpha + beta_param)
            
            # Simple distance metric
            return float(1.0 - abs(new_observation - mean))
        
        elif posterior.distribution == "gamma":
            alpha = posterior.get_param("alpha", 1.0)
            beta_param = posterior.get_param("beta", 1.0)
            
            # Predictive mean (for Poisson)
            mean = alpha / beta_param
            
            # Simple distance metric
            return float(np.exp(-abs(new_observation - mean)))
        
        return 0.5
