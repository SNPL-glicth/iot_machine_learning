"""Activation functions for classical feedforward layer.

Pure numpy implementations — no external dependencies.
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation.
    
    f(x) = max(0, x)
    
    Args:
        x: Input array
        
    Returns:
        Activated array
    """
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation.
    
    f(x) = 1 / (1 + exp(-x))
    
    Args:
        x: Input array
        
    Returns:
        Activated array (values in [0, 1])
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation (for output layer).
    
    f(x_i) = exp(x_i) / sum(exp(x_j))
    
    Numerically stable implementation.
    
    Args:
        x: Input array (1D or 2D)
        
    Returns:
        Probability distribution (sums to 1)
    """
    # Numerical stability: subtract max
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation.
    
    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        x: Input array
        
    Returns:
        Activated array (values in [-1, 1])
    """
    return np.tanh(x)


def linear(x: np.ndarray) -> np.ndarray:
    """Linear activation (identity).
    
    f(x) = x
    
    Args:
        x: Input array
        
    Returns:
        Unchanged array
    """
    return x
