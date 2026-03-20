"""Maximum Likelihood Estimator — unified interface."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from .distributions import (
    GaussianMLE,
    PoissonMLE,
    BetaMLE,
    ExponentialMLE,
)


@dataclass
class MLEResult:
    """Maximum likelihood estimation result."""
    distribution: str
    parameters: Dict[str, float]
    log_likelihood: float
    n_samples: int
    
    def get_param(self, name: str, default: float = 0.0) -> float:
        """Get parameter by name with fallback."""
        return self.parameters.get(name, default)


class MaximumLikelihoodEstimator:
    """Unified MLE interface for common distributions.
    
    Supported distributions:
    - gaussian: (μ, σ²)
    - poisson: (λ,)
    - beta: (α, β)
    - exponential: (λ,)
    
    Example:
        mle = MaximumLikelihoodEstimator()
        result = mle.fit(data, distribution="gaussian")
        mu = result.get_param("mu")
        sigma2 = result.get_param("sigma2")
    """
    
    def __init__(self):
        self._fitters = {
            "gaussian": GaussianMLE.fit,
            "poisson": PoissonMLE.fit,
            "beta": BetaMLE.fit,
            "exponential": ExponentialMLE.fit,
        }
    
    def fit(
        self,
        data: np.ndarray,
        distribution: str = "gaussian",
    ) -> MLEResult:
        """Fit distribution to data via MLE.
        
        Args:
            data: Observations (1D array)
            distribution: Distribution type
            
        Returns:
            MLEResult with fitted parameters
            
        Raises:
            ValueError: If distribution not supported
        """
        if distribution not in self._fitters:
            raise ValueError(
                f"Unsupported distribution: {distribution}. "
                f"Must be one of {list(self._fitters.keys())}"
            )
        
        # Ensure numpy array
        data_array = np.asarray(data).flatten()
        
        # Fit via distribution-specific MLE
        fitter = self._fitters[distribution]
        params_obj = fitter(data_array)
        
        return MLEResult(
            distribution=distribution,
            parameters=params_obj.params,
            log_likelihood=params_obj.log_likelihood,
            n_samples=params_obj.n_samples,
        )
    
    def fit_best(
        self,
        data: np.ndarray,
        candidates: list[str] = None,
    ) -> MLEResult:
        """Fit all candidates, return best by log-likelihood.
        
        Args:
            data: Observations
            candidates: List of distributions to try (default: all)
            
        Returns:
            MLEResult for best-fitting distribution
        """
        if candidates is None:
            candidates = list(self._fitters.keys())
        
        results = []
        for dist in candidates:
            try:
                result = self.fit(data, distribution=dist)
                results.append(result)
            except Exception:
                # Skip failed fits
                continue
        
        if not results:
            # Fallback to Gaussian
            return self.fit(data, distribution="gaussian")
        
        # Return distribution with highest log-likelihood
        best = max(results, key=lambda r: r.log_likelihood)
        return best
