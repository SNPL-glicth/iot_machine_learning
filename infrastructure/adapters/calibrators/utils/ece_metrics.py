"""Expected Calibration Error (ECE) computation utilities.
"""

from typing import Dict, Tuple

import numpy as np


def compute_ece_numpy(
    calibrated: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict[int, Tuple[float, float]]]:
    """Compute ECE using numpy arrays.
    
    Args:
        calibrated: Array of calibrated scores [0, 1].
        outcomes: Array of actual outcomes (0.0 or 1.0).
        n_bins: Number of bins for ECE computation.
        
    Returns:
        Tuple of (ece_value, reliability_dict).
    """
    if len(calibrated) < 10:
        return 0.0, {}
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability: Dict[int, Tuple[float, float]] = {}
    
    for i in range(n_bins):
        mask = (calibrated >= bin_edges[i]) & (calibrated < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (calibrated >= bin_edges[i]) & (calibrated <= bin_edges[i + 1])
        
        bin_scores = calibrated[mask]
        bin_outcomes = outcomes[mask]
        
        if len(bin_scores) > 0:
            avg_confidence = float(np.mean(bin_scores))
            avg_accuracy = float(np.mean(bin_outcomes))
            bin_weight = len(bin_scores) / len(calibrated)
            
            ece += abs(avg_confidence - avg_accuracy) * bin_weight
            reliability[i] = (avg_confidence, avg_accuracy)
    
    return float(ece), reliability


def compute_ece(
    scores: list[float],
    outcomes: list[float],
    n_bins: int = 10,
) -> Tuple[float, Dict[int, Tuple[float, float]]]:
    """Compute ECE from Python lists.
    
    Args:
        scores: List of calibrated confidence scores.
        outcomes: List of actual outcomes (0.0 or 1.0).
        n_bins: Number of bins for ECE computation.
        
    Returns:
        Tuple of (ece_value, reliability_dict).
    """
    if len(scores) < 10:
        return 0.0, {}
    
    return compute_ece_numpy(np.array(scores), np.array(outcomes), n_bins)
