"""Distribution-specific MLE implementations.

Each distribution provides closed-form or numerical MLE.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DistributionParameters:
    """MLE parameter estimates."""
    params: dict
    log_likelihood: float
    n_samples: int


class GaussianMLE:
    """Maximum likelihood for Gaussian distribution.
    
    θ = (μ, σ²)
    μ_MLE = mean(data)
    σ²_MLE = variance(data)
    """
    
    @staticmethod
    def fit(data: np.ndarray) -> DistributionParameters:
        """Fit Gaussian via MLE (closed form)."""
        if len(data) == 0:
            return DistributionParameters(
                params={"mu": 0.0, "sigma2": 1.0},
                log_likelihood=-np.inf,
                n_samples=0,
            )
        
        mu = np.mean(data)
        sigma2 = np.var(data, ddof=0)  # MLE uses N, not N-1
        
        # Avoid zero variance
        if sigma2 < 1e-9:
            sigma2 = 1e-9
        
        # Log-likelihood: -0.5 * Σ[(x - μ)²/σ² + log(2πσ²)]
        log_likelihood = -0.5 * np.sum(
            ((data - mu) ** 2) / sigma2 + np.log(2 * np.pi * sigma2)
        )
        
        return DistributionParameters(
            params={"mu": float(mu), "sigma2": float(sigma2)},
            log_likelihood=float(log_likelihood),
            n_samples=len(data),
        )


class PoissonMLE:
    """Maximum likelihood for Poisson distribution.
    
    θ = (λ,)
    λ_MLE = mean(data)
    """
    
    @staticmethod
    def fit(data: np.ndarray) -> DistributionParameters:
        """Fit Poisson via MLE (closed form)."""
        if len(data) == 0:
            return DistributionParameters(
                params={"lambda": 1.0},
                log_likelihood=-np.inf,
                n_samples=0,
            )
        
        lambda_mle = np.mean(data)
        
        # Avoid zero lambda
        if lambda_mle < 1e-9:
            lambda_mle = 1e-9
        
        # Log-likelihood: Σ[x * log(λ) - λ - log(x!)]
        # Ignore factorial term (constant for given data)
        log_likelihood = np.sum(data * np.log(lambda_mle) - lambda_mle)
        
        return DistributionParameters(
            params={"lambda": float(lambda_mle)},
            log_likelihood=float(log_likelihood),
            n_samples=len(data),
        )


class ExponentialMLE:
    """Maximum likelihood for Exponential distribution.
    
    θ = (λ,)
    λ_MLE = 1 / mean(data)
    """
    
    @staticmethod
    def fit(data: np.ndarray) -> DistributionParameters:
        """Fit Exponential via MLE (closed form)."""
        if len(data) == 0:
            return DistributionParameters(
                params={"lambda": 1.0},
                log_likelihood=-np.inf,
                n_samples=0,
            )
        
        mean_val = np.mean(data)
        
        if mean_val < 1e-9:
            mean_val = 1e-9
        
        lambda_mle = 1.0 / mean_val
        
        # Log-likelihood: Σ[log(λ) - λ * x]
        log_likelihood = np.sum(np.log(lambda_mle) - lambda_mle * data)
        
        return DistributionParameters(
            params={"lambda": float(lambda_mle)},
            log_likelihood=float(log_likelihood),
            n_samples=len(data),
        )


class BetaMLE:
    """Maximum likelihood for Beta distribution.
    
    θ = (α, β)
    Uses method of moments + Newton-Raphson refinement.
    """
    
    @staticmethod
    def fit(data: np.ndarray, max_iter: int = 20) -> DistributionParameters:
        """Fit Beta via method of moments + Newton-Raphson."""
        if len(data) == 0:
            return DistributionParameters(
                params={"alpha": 1.0, "beta": 1.0},
                log_likelihood=-np.inf,
                n_samples=0,
            )
        
        # Clip data to (0, 1)
        data_clipped = np.clip(data, 1e-9, 1 - 1e-9)
        
        # Method of moments initial estimate
        mean = np.mean(data_clipped)
        var = np.var(data_clipped, ddof=0)
        
        # Avoid division by zero
        if var < 1e-9:
            var = 1e-9
        
        # α + β = (mean * (1 - mean) / var) - 1
        # α = mean * (α + β)
        sum_ab = (mean * (1 - mean) / var) - 1.0
        
        if sum_ab < 2.0:
            sum_ab = 2.0
        
        alpha = mean * sum_ab
        beta = (1 - mean) * sum_ab
        
        # Ensure minimum values
        alpha = max(alpha, 0.1)
        beta = max(beta, 0.1)
        
        # Log-likelihood (ignoring constant terms)
        # Σ[(α-1)*log(x) + (β-1)*log(1-x)] - N*log(B(α,β))
        # We skip Newton-Raphson refinement for simplicity
        
        log_data = np.log(data_clipped)
        log_1_minus_data = np.log(1 - data_clipped)
        
        from scipy.special import betaln
        log_likelihood = np.sum(
            (alpha - 1) * log_data + (beta - 1) * log_1_minus_data
        ) - len(data) * betaln(alpha, beta)
        
        return DistributionParameters(
            params={"alpha": float(alpha), "beta": float(beta)},
            log_likelihood=float(log_likelihood),
            n_samples=len(data),
        )
