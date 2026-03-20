"""Proximal gradient optimization for regularized problems.

Handles non-smooth regularization terms (L1, L2) efficiently.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Literal
from ..types import OptimizationResult, OptimizerConfig


class ProximalGradientOptimizer:
    """Proximal gradient descent for composite optimization.
    
    Solves: minimize f(x) + λ * g(x)
    where f is smooth and g is convex but possibly non-smooth.
    
    Update: x_{k+1} = prox_{λg}(x_k - α * ∇f(x_k))
    
    Supports:
    - L1 regularization (Lasso): g(x) = ||x||_1
    - L2 regularization (Ridge): g(x) = ||x||_2²
    
    Args:
        config: Optimizer configuration
        regularization: Regularization type ('l1' or 'l2')
        lambda_reg: Regularization strength
        lr: Learning rate
    """
    
    def __init__(
        self,
        config: OptimizerConfig = None,
        regularization: Literal['l1', 'l2'] = 'l1',
        lambda_reg: float = 0.01,
        lr: float = 0.01,
    ):
        self.config = config or OptimizerConfig()
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.lr = lr
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
    ) -> OptimizationResult:
        """Optimize using proximal gradient.
        
        Args:
            objective: Smooth part of objective f(x)
            gradient_fn: Gradient of smooth part ∇f(x)
            initial_params: Starting parameters
            
        Returns:
            OptimizationResult
        """
        params = initial_params.copy()
        history = {"objective": [], "norm_gradient": []}
        
        for iteration in range(self.config.max_iterations):
            # Gradient of smooth part
            gradient = gradient_fn(params)
            
            # Gradient step
            params_intermediate = params - self.lr * gradient
            
            # Proximal operator (handles regularization)
            params_new = self._proximal_operator(params_intermediate, self.lr * self.lambda_reg)
            
            # Evaluate full objective (smooth + regularization)
            obj_value = objective(params_new)
            if self.regularization == 'l1':
                obj_value += self.lambda_reg * np.sum(np.abs(params_new))
            elif self.regularization == 'l2':
                obj_value += 0.5 * self.lambda_reg * np.sum(params_new ** 2)
            
            history["objective"].append(obj_value)
            history["norm_gradient"].append(float(np.linalg.norm(gradient)))
            
            # Check convergence
            param_change = np.linalg.norm(params_new - params)
            if param_change < self.config.tolerance:
                return OptimizationResult(
                    params=params_new,
                    objective_value=obj_value,
                    success=True,
                    n_iterations=iteration + 1,
                    message="Converged",
                    history=history,
                )
            
            params = params_new
        
        return OptimizationResult(
            params=params,
            objective_value=objective(params),
            success=False,
            n_iterations=self.config.max_iterations,
            message="Max iterations reached",
            history=history,
        )
    
    def _proximal_operator(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Apply proximal operator for regularization.
        
        Args:
            x: Input
            threshold: Regularization threshold
            
        Returns:
            Proximal output
        """
        if self.regularization == 'l1':
            # Soft thresholding: prox(x) = sign(x) * max(|x| - threshold, 0)
            return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        
        elif self.regularization == 'l2':
            # Scaling: prox(x) = x / (1 + threshold)
            return x / (1 + threshold)
        
        else:
            return x
