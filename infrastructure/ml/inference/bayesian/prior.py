"""Prior distributions for Bayesian inference.

Conjugate priors for common likelihoods:
- Gaussian → Gaussian prior (normal-normal)
- Beta → Beta prior (beta-bernoulli)
- Poisson → Gamma prior (gamma-poisson)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Prior:
    """Base prior distribution."""
    distribution: str
    parameters: Dict[str, float]
    
    def get_param(self, name: str, default: float = 0.0) -> float:
        """Get parameter by name."""
        return self.parameters.get(name, default)


@dataclass
class GaussianPrior(Prior):
    """Gaussian prior for Gaussian likelihood.
    
    N(μ_0, σ²_0)
    Used for: mean estimation with Gaussian data
    """
    
    def __init__(self, mu_0: float = 0.0, sigma2_0: float = 1.0):
        super().__init__(
            distribution="gaussian",
            parameters={"mu_0": mu_0, "sigma2_0": sigma2_0},
        )


@dataclass
class BetaPrior(Prior):
    """Beta prior for Bernoulli/binomial likelihood.
    
    Beta(α, β)
    Used for: probability estimation
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(
            distribution="beta",
            parameters={"alpha": alpha, "beta": beta},
        )


@dataclass
class GammaPrior(Prior):
    """Gamma prior for Poisson/exponential likelihood.
    
    Gamma(α, β)
    Used for: rate estimation
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(
            distribution="gamma",
            parameters={"alpha": alpha, "beta": beta},
        )
