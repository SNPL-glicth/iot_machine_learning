"""Noise model for Monte Carlo perturbations.

Input-type dependent noise standard deviations and perturbation logic.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..types import InputType

# Noise model: standard deviation per input type
NOISE_SIGMA_TEXT = 0.10      # Keyword detection ~10% variance
NOISE_SIGMA_NUMERIC = 0.05   # Structural scores more precise
NOISE_SIGMA_MIXED = 0.075    # Average of text and numeric


def get_noise_sigma(input_type: InputType) -> float:
    """Get noise standard deviation for input type.
    
    Args:
        input_type: Detected input type
        
    Returns:
        Sigma value for Gaussian noise
    """
    if input_type == InputType.TEXT:
        return NOISE_SIGMA_TEXT
    elif input_type == InputType.NUMERIC:
        return NOISE_SIGMA_NUMERIC
    elif input_type == InputType.MIXED:
        return NOISE_SIGMA_MIXED
    else:
        # Default to text sigma for unknown types
        return NOISE_SIGMA_TEXT


def perturb_scores(
    scores: Dict[str, float],
    sigma: float,
) -> Dict[str, float]:
    """Add Gaussian noise to analysis scores.
    
    Args:
        scores: Original scores
        sigma: Standard deviation of noise
        
    Returns:
        Perturbed scores (clamped to [0, 1])
    """
    perturbed = {}
    for key, value in scores.items():
        # Add Gaussian noise
        noise = np.random.normal(0, sigma)
        perturbed_value = value + noise
        
        # Clamp to valid range [0, 1]
        perturbed[key] = max(0.0, min(1.0, perturbed_value))
    
    return perturbed
