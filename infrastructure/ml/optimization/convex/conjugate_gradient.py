"""Conjugate gradient optimization.

Efficient for large-scale convex quadratic problems.
Avoids computing full Hessian matrix.
"""

from __future__ import annotations

import numpy as np
from typing import Callable
from ..types import OptimizationResult, OptimizerConfig


class ConjugateGradientOptimizer:
    """Conjugate gradient method for quadratic optimization.
    
    Solves: minimize 0.5 * x^T A x - b^T x
    without explicitly forming matrix A.
    
    Uses conjugate directions instead of steepest descent.
    Converges in at most n iterations for n-dimensional problem.
    
    Args:
        config: Optimizer configuration
    """
    
    def __init__(self, config: OptimizerConfig = None):
        self.config = config or OptimizerConfig()
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
    ) -> OptimizationResult:
        """Optimize using conjugate gradient (Polak-Ribière variant).
        
        Args:
            objective: Objective function
            gradient_fn: Gradient function
            initial_params: Starting parameters
            
        Returns:
            OptimizationResult
        """
        params = initial_params.copy()
        gradient = gradient_fn(params)
        
        # Initial search direction (negative gradient)
        direction = -gradient
        
        history = {"objective": [], "norm_gradient": []}
        
        for iteration in range(self.config.max_iterations):
            obj_value = objective(params)
            grad_norm = float(np.linalg.norm(gradient))
            
            history["objective"].append(obj_value)
            history["norm_gradient"].append(grad_norm)
            
            # Check convergence
            if grad_norm < self.config.tolerance:
                return OptimizationResult(
                    params=params,
                    objective_value=obj_value,
                    success=True,
                    n_iterations=iteration + 1,
                    message="Converged",
                    history=history,
                )
            
            # Line search for step size
            alpha = self._line_search(objective, gradient_fn, params, direction, gradient)
            
            # Update parameters
            params_new = params + alpha * direction
            gradient_new = gradient_fn(params_new)
            
            # Polak-Ribière formula for β
            gradient_diff = gradient_new - gradient
            beta = max(0, (gradient_new @ gradient_diff) / (gradient @ gradient + 1e-10))
            
            # Update direction (conjugate to previous)
            direction = -gradient_new + beta * direction
            
            params = params_new
            gradient = gradient_new
        
        return OptimizationResult(
            params=params,
            objective_value=objective(params),
            success=False,
            n_iterations=self.config.max_iterations,
            message="Max iterations reached",
            history=history,
        )
    
    def _line_search(
        self,
        objective: Callable,
        gradient_fn: Callable,
        params: np.ndarray,
        direction: np.ndarray,
        gradient: np.ndarray,
    ) -> float:
        """Simple backtracking line search.
        
        Args:
            objective: Objective function
            gradient_fn: Gradient function
            params: Current parameters
            direction: Search direction
            gradient: Current gradient
            
        Returns:
            Step size alpha
        """
        alpha = 1.0
        obj_current = objective(params)
        c1 = 1e-4
        directional_derivative = gradient @ direction
        
        for _ in range(20):
            params_new = params + alpha * direction
            obj_new = objective(params_new)
            
            # Armijo condition
            if obj_new <= obj_current + c1 * alpha * directional_derivative:
                return alpha
            
            alpha *= 0.5
        
        return alpha
