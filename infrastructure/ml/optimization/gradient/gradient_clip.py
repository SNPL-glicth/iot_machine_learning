"""Gradient clipping utilities.

Prevents exploding gradients in deep networks or RNNs.
Two main strategies:
- Clip by value: g = clip(g, -threshold, threshold)
- Clip by norm: g = g * min(1, threshold / ||g||)
"""

from __future__ import annotations

import numpy as np


def compute_gradient_norm(gradients: np.ndarray, ord: int = 2) -> float:
    """Compute gradient norm.
    
    Args:
        gradients: Gradient array
        ord: Norm order (1, 2, inf)
        
    Returns:
        Gradient norm
    """
    if ord == 1:
        return float(np.sum(np.abs(gradients)))
    elif ord == 2:
        return float(np.sqrt(np.sum(gradients ** 2)))
    elif ord == np.inf:
        return float(np.max(np.abs(gradients)))
    else:
        raise ValueError(f"Unsupported norm order: {ord}")


def clip_gradients(
    gradients: np.ndarray,
    clip_value: float = None,
    clip_norm: float = None,
) -> np.ndarray:
    """Clip gradients by value or norm.
    
    Args:
        gradients: Gradients to clip
        clip_value: Clip by value (±clip_value)
        clip_norm: Clip by L2 norm
        
    Returns:
        Clipped gradients
        
    Raises:
        ValueError: If both clip_value and clip_norm specified
    """
    if clip_value is not None and clip_norm is not None:
        raise ValueError("Cannot specify both clip_value and clip_norm")
    
    if clip_value is not None:
        # Clip by value
        return np.clip(gradients, -clip_value, clip_value)
    
    elif clip_norm is not None:
        # Clip by norm
        norm = compute_gradient_norm(gradients, ord=2)
        
        if norm > clip_norm:
            # Scale down: g = g * (clip_norm / ||g||)
            return gradients * (clip_norm / norm)
        else:
            return gradients
    
    else:
        # No clipping
        return gradients
