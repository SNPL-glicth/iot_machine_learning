"""Platt Scaling mathematical utilities.
"""

from typing import Tuple

import numpy as np


def platt_sigmoid(scores: np.ndarray, A: float, B: float) -> np.ndarray:
    """Apply Platt sigmoid: P = 1 / (1 + exp(A * score + B)).
    
    Args:
        scores: Array of raw confidence scores.
        A: Sigmoid parameter (slope).
        B: Sigmoid parameter (intercept).
        
    Returns:
        Array of calibrated probabilities.
    """
    logits = A * scores + B
    logits = np.clip(logits, -500, 500)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(logits))


def fit_platt_params(
    scores: np.ndarray,
    outcomes: np.ndarray,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> Tuple[float, float]:
    """Fit Platt scaling parameters using Newton-Raphson.
    
    Args:
        scores: Array of raw confidence scores.
        outcomes: Array of actual outcomes (0.0 or 1.0).
        max_iterations: Maximum Newton-Raphson iterations.
        tolerance: Convergence tolerance for parameter changes.
        
    Returns:
        Tuple of (A, B) parameters.
    """
    A, B = 0.0, 0.0  # Neutral initialization
    
    for _ in range(max_iterations):
        logits = A * scores + B
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        
        # Gradients
        dA = np.sum(scores * (probs - outcomes))
        dB = np.sum(probs - outcomes)
        
        # Hessians
        H_aa = np.sum(scores**2 * probs * (1 - probs))
        H_bb = np.sum(probs * (1 - probs))
        H_ab = np.sum(scores * probs * (1 - probs))
        
        # Solve linear system
        det = H_aa * H_bb - H_ab**2
        if abs(det) < 1e-10:
            break
        
        delta_A = (H_bb * dA - H_ab * dB) / det
        delta_B = (-H_ab * dA + H_aa * dB) / det
        
        A -= delta_A
        B -= delta_B
        
        if abs(delta_A) < tolerance and abs(delta_B) < tolerance:
            break
    
    return float(A), float(B)
