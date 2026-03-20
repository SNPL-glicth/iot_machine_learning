"""Statistical computation utilities for Monte Carlo analysis.

Distribution, confidence intervals, and uncertainty classification.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Uncertainty level thresholds (based on std of severity scores)
UNCERTAINTY_LOW_THRESHOLD = 0.10
UNCERTAINTY_HIGH_THRESHOLD = 0.25


def compute_distribution(
    severity_samples: List[str],
) -> Dict[str, float]:
    """Compute probability distribution over severity levels.
    
    Args:
        severity_samples: List of severity labels from simulations
        
    Returns:
        Dict mapping severity → probability
    """
    total = len(severity_samples)
    counts = {}
    
    for severity in severity_samples:
        counts[severity] = counts.get(severity, 0) + 1
    
    # Normalize to probabilities
    distribution = {
        severity: count / total
        for severity, count in counts.items()
    }
    
    return distribution


def compute_confidence_interval(
    severity_scores: List[float],
) -> Tuple[float, float]:
    """Compute 95% confidence interval.
    
    Args:
        severity_scores: Numeric severity scores [0, 1]
        
    Returns:
        Tuple (lower_bound, upper_bound) at 95% confidence
    """
    scores_array = np.array(severity_scores)
    
    lower = float(np.percentile(scores_array, 5))
    upper = float(np.percentile(scores_array, 95))
    
    return (lower, upper)


def classify_uncertainty(
    severity_scores: List[float],
) -> str:
    """Classify uncertainty level based on score variance.
    
    Args:
        severity_scores: Numeric severity scores
        
    Returns:
        "low" | "moderate" | "high"
    """
    std = float(np.std(severity_scores))
    
    if std < UNCERTAINTY_LOW_THRESHOLD:
        return "low"
    elif std < UNCERTAINTY_HIGH_THRESHOLD:
        return "moderate"
    else:
        return "high"
