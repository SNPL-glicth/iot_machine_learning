"""Convenience function for one-line MLE fitting."""

from __future__ import annotations

import numpy as np
from typing import Dict

from .estimator import MaximumLikelihoodEstimator, MLEResult


def fit_distribution(
    data: np.ndarray,
    distribution: str = "gaussian",
) -> Dict[str, float]:
    """Fit distribution to data, return parameters dict.
    
    Convenience wrapper around MaximumLikelihoodEstimator.
    
    Args:
        data: Observations
        distribution: Distribution type
        
    Returns:
        Dict of fitted parameters
        
    Example:
        params = fit_distribution([1.2, 1.5, 1.3], "gaussian")
        # params = {"mu": 1.33, "sigma2": 0.02}
    """
    estimator = MaximumLikelihoodEstimator()
    result = estimator.fit(data, distribution)
    return result.parameters
