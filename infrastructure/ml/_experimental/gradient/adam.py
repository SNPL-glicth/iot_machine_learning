"""Adaptive moment estimation optimizers.

Adam: Combines RMSProp + Momentum with bias correction
AdaGrad: Adapts learning rate per parameter
RMSProp: Uses moving average of squared gradients
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class AdamOptimizer:
    """Adaptive Moment Estimation (Adam).
    
    Combines momentum + RMSProp with bias correction:
        m_t = β1 * m_{t-1} + (1-β1) * g_t      (first moment)
        v_t = β2 * v_{t-1} + (1-β2) * g_t²     (second moment)
        m̂_t = m_t / (1 - β1^t)                 (bias correction)
        v̂_t = v_t / (1 - β2^t)
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    
    Args:
        lr: Learning rate (default: 0.001)
        beta1: First moment decay (default: 0.9)
        beta2: Second moment decay (default: 0.999)
        epsilon: Small constant for numerical stability
        weight_decay: L2 penalty coefficient
    """
    
    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m: Optional[np.ndarray] = None  # First moment
        self.v: Optional[np.ndarray] = None  # Second moment
        self.t = 0
    
    def step(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """Single optimization step.
        
        Args:
            params: Current parameters
            gradients: Gradients at current params
            
        Returns:
            Updated parameters
        """
        self.t += 1
        
        # Initialize moments on first step
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # Apply weight decay
        if self.weight_decay > 0:
            gradients = gradients + self.weight_decay * params
        
        # Update biased first moment: m = β1 * m + (1-β1) * g
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second moment: v = β2 * v + (1-β2) * g²
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters: θ = θ - α * m̂ / (√v̂ + ε)
        updated_params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.t = 0


class AdaGradOptimizer:
    """Adaptive Gradient Algorithm (AdaGrad).
    
    Adapts learning rate per parameter based on historical gradients:
        G_t = G_{t-1} + g_t²
        θ_t = θ_{t-1} - α / (√G_t + ε) * g_t
    
    Good for sparse data, but learning rate can become too small.
    
    Args:
        lr: Learning rate
        epsilon: Small constant for numerical stability
        weight_decay: L2 penalty coefficient
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.G: Optional[np.ndarray] = None  # Accumulated squared gradients
        self.t = 0
    
    def step(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """Single optimization step.
        
        Args:
            params: Current parameters
            gradients: Gradients at current params
            
        Returns:
            Updated parameters
        """
        self.t += 1
        
        # Initialize accumulator on first step
        if self.G is None:
            self.G = np.zeros_like(params)
        
        # Apply weight decay
        if self.weight_decay > 0:
            gradients = gradients + self.weight_decay * params
        
        # Accumulate squared gradients: G = G + g²
        self.G += gradients ** 2
        
        # Adaptive learning rate: α_adapted = α / √(G + ε)
        adapted_lr = self.lr / (np.sqrt(self.G) + self.epsilon)
        
        # Update parameters: θ = θ - α_adapted * g
        updated_params = params - adapted_lr * gradients
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state."""
        self.G = None
        self.t = 0


class RMSPropOptimizer:
    """Root Mean Square Propagation (RMSProp).
    
    Uses moving average of squared gradients (fixes AdaGrad's learning rate decay):
        E[g²]_t = β * E[g²]_{t-1} + (1-β) * g_t²
        θ_t = θ_{t-1} - α / (√E[g²]_t + ε) * g_t
    
    Args:
        lr: Learning rate
        rho: Decay rate for moving average (typically 0.9)
        epsilon: Small constant for numerical stability
        weight_decay: L2 penalty coefficient
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        rho: float = 0.9,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.E_g2: Optional[np.ndarray] = None  # Moving average of g²
        self.t = 0
    
    def step(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """Single optimization step.
        
        Args:
            params: Current parameters
            gradients: Gradients at current params
            
        Returns:
            Updated parameters
        """
        self.t += 1
        
        # Initialize moving average on first step
        if self.E_g2 is None:
            self.E_g2 = np.zeros_like(params)
        
        # Apply weight decay
        if self.weight_decay > 0:
            gradients = gradients + self.weight_decay * params
        
        # Update moving average: E[g²] = ρ * E[g²] + (1-ρ) * g²
        self.E_g2 = self.rho * self.E_g2 + (1 - self.rho) * (gradients ** 2)
        
        # Adaptive learning rate: α_adapted = α / √(E[g²] + ε)
        adapted_lr = self.lr / (np.sqrt(self.E_g2) + self.epsilon)
        
        # Update parameters: θ = θ - α_adapted * g
        updated_params = params - adapted_lr * gradients
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state."""
        self.E_g2 = None
        self.t = 0
