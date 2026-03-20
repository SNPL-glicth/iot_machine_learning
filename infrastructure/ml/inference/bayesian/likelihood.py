"""Likelihood functions for Bayesian inference.

Each likelihood corresponds to a data-generating distribution.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


def gaussian_likelihood(
    data: np.ndarray,
    mu: float,
    sigma2: float,
) -> float:
    """Gaussian likelihood: P(data | μ, σ²).
    
    Args:
        data: Observations
        mu: Mean parameter
        sigma2: Variance parameter
        
    Returns:
        Log-likelihood
    """
    if sigma2 <= 0:
        sigma2 = 1e-9
    
    log_likelihood = -0.5 * np.sum(
        ((data - mu) ** 2) / sigma2 + np.log(2 * np.pi * sigma2)
    )
    
    return float(log_likelihood)


def beta_likelihood(
    data: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Beta likelihood: P(data | α, β).
    
    Args:
        data: Observations in [0, 1]
        alpha: Alpha parameter
        beta: Beta parameter
        
    Returns:
        Log-likelihood
    """
    from scipy.special import betaln
    
    data_clipped = np.clip(data, 1e-9, 1 - 1e-9)
    
    log_likelihood = np.sum(
        (alpha - 1) * np.log(data_clipped) +
        (beta - 1) * np.log(1 - data_clipped)
    ) - len(data) * betaln(alpha, beta)
    
    return float(log_likelihood)


def poisson_likelihood(
    data: np.ndarray,
    lambda_param: float,
) -> float:
    """Poisson likelihood: P(data | λ).
    
    Args:
        data: Count observations
        lambda_param: Rate parameter
        
    Returns:
        Log-likelihood
    """
    if lambda_param <= 0:
        lambda_param = 1e-9
    
    # Ignore factorial term (constant for given data)
    log_likelihood = np.sum(
        data * np.log(lambda_param) - lambda_param
    )
    
    return float(log_likelihood)


def get_likelihood_function(distribution: str) -> Callable:
    """Get likelihood function by distribution name.
    
    Args:
        distribution: Distribution type
        
    Returns:
        Likelihood function
    """
    likelihoods = {
        "gaussian": gaussian_likelihood,
        "beta": beta_likelihood,
        "poisson": poisson_likelihood,
    }
    
    if distribution not in likelihoods:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return likelihoods[distribution]
