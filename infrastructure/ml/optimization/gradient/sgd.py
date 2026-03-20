"""Stochastic Gradient Descent optimizers.

SGD variants:
- Vanilla SGD: θ_{t+1} = θ_t - α * g_t
- Momentum: v_t = β * v_{t-1} + g_t, θ_{t+1} = θ_t - α * v_t
- Nesterov: v_t = β * v_{t-1} + g(θ_t - α * β * v_{t-1})
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class SGDOptimizer:
    """Vanilla stochastic gradient descent.
    
    Update rule: θ_{t+1} = θ_t - α * g_t
    
    Args:
        lr: Learning rate
        weight_decay: L2 penalty coefficient
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
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
        
        # Apply weight decay (L2 regularization)
        if self.weight_decay > 0:
            gradients = gradients + self.weight_decay * params
        
        # Update: θ = θ - α * g
        updated_params = params - self.lr * gradients
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state."""
        self.t = 0


class MomentumSGD:
    """SGD with momentum.
    
    Update rule:
        v_t = β * v_{t-1} + g_t
        θ_{t+1} = θ_t - α * v_t
    
    Momentum accelerates convergence and dampens oscillations.
    
    Args:
        lr: Learning rate
        momentum: Momentum coefficient (typically 0.9)
        weight_decay: L2 penalty coefficient
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: Optional[np.ndarray] = None
        self.t = 0
    
    def step(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """Single optimization step with momentum.
        
        Args:
            params: Current parameters
            gradients: Gradients at current params
            
        Returns:
            Updated parameters
        """
        self.t += 1
        
        # Initialize velocity on first step
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # Apply weight decay
        if self.weight_decay > 0:
            gradients = gradients + self.weight_decay * params
        
        # Update velocity: v = β * v + g
        self.velocity = self.momentum * self.velocity + gradients
        
        # Update params: θ = θ - α * v
        updated_params = params - self.lr * self.velocity
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state."""
        self.velocity = None
        self.t = 0


class NesterovSGD:
    """SGD with Nesterov momentum.
    
    Nesterov accelerated gradient (NAG) looks ahead:
        v_t = β * v_{t-1} + ∇f(θ_t - α * β * v_{t-1})
        θ_{t+1} = θ_t - α * v_t
    
    More responsive than standard momentum.
    
    Args:
        lr: Learning rate
        momentum: Momentum coefficient (typically 0.9)
        weight_decay: L2 penalty coefficient
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: Optional[np.ndarray] = None
        self.t = 0
    
    def step(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """Single optimization step with Nesterov momentum.
        
        Note: This is the simplified implementation where the gradient
        is already computed at the look-ahead point.
        
        Args:
            params: Current parameters
            gradients: Gradients (should be computed at look-ahead point)
            
        Returns:
            Updated parameters
        """
        self.t += 1
        
        # Initialize velocity on first step
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # Apply weight decay
        if self.weight_decay > 0:
            gradients = gradients + self.weight_decay * params
        
        # Update velocity: v = β * v + g
        self.velocity = self.momentum * self.velocity + gradients
        
        # Update params: θ = θ - α * v
        updated_params = params - self.lr * self.velocity
        
        return updated_params
    
    def get_lookahead_params(self, params: np.ndarray) -> np.ndarray:
        """Get look-ahead parameters for gradient computation.
        
        Args:
            params: Current parameters
            
        Returns:
            Look-ahead parameters
        """
        if self.velocity is None:
            return params
        
        # Look ahead: θ_lookahead = θ - α * β * v
        return params - self.lr * self.momentum * self.velocity
    
    def reset(self):
        """Reset optimizer state."""
        self.velocity = None
        self.t = 0
