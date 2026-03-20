"""Limited-memory BFGS optimizer.

L-BFGS approximates inverse Hessian using only last m gradient pairs.
Memory efficient for high-dimensional problems.
"""

from __future__ import annotations

import numpy as np
from typing import Callable
from collections import deque
from ..types import OptimizationResult, OptimizerConfig


class LBFGSOptimizer:
    """Limited-memory BFGS optimizer.
    
    Memory-efficient quasi-Newton method for large-scale optimization.
    Stores only last m (s, y) pairs instead of full inverse Hessian.
    
    Args:
        config: Optimizer configuration
        m: Number of correction pairs to store (default: 10)
        line_search_max_iter: Max iterations for line search
    """
    
    def __init__(
        self,
        config: OptimizerConfig = None,
        m: int = 10,
        line_search_max_iter: int = 20,
    ):
        self.config = config or OptimizerConfig()
        self.m = m
        self.line_search_max_iter = line_search_max_iter
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
    ) -> OptimizationResult:
        """Optimize using L-BFGS.
        
        Args:
            objective: Objective function
            gradient_fn: Gradient function
            initial_params: Starting parameters
            
        Returns:
            OptimizationResult
        """
        params = initial_params.copy()
        gradient = gradient_fn(params)
        
        # Storage for (s, y) pairs
        s_history = deque(maxlen=self.m)
        y_history = deque(maxlen=self.m)
        rho_history = deque(maxlen=self.m)
        
        history = {"objective": [], "norm_gradient": []}
        
        for iteration in range(self.config.max_iterations):
            obj_value = objective(params)
            history["objective"].append(obj_value)
            grad_norm = float(np.linalg.norm(gradient))
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
            
            # Compute search direction using L-BFGS two-loop recursion
            direction = self._compute_direction(gradient, s_history, y_history, rho_history)
            
            # Line search
            alpha = self._line_search(objective, gradient_fn, params, direction, gradient)
            
            # Update parameters
            params_new = params + alpha * direction
            gradient_new = gradient_fn(params_new)
            
            # Store correction pair
            s = params_new - params
            y = gradient_new - gradient
            
            y_dot_s = y @ s
            
            if y_dot_s > 1e-10:  # Ensure positive curvature
                s_history.append(s)
                y_history.append(y)
                rho_history.append(1.0 / y_dot_s)
            
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
    
    def _compute_direction(
        self,
        gradient: np.ndarray,
        s_history: deque,
        y_history: deque,
        rho_history: deque,
    ) -> np.ndarray:
        """Compute search direction using L-BFGS two-loop recursion."""
        if len(s_history) == 0:
            # No history → use steepest descent
            return -gradient
        
        q = gradient.copy()
        alpha_values = []
        
        # First loop (backward)
        for s, y, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
            alpha = rho * (s @ q)
            alpha_values.append(alpha)
            q = q - alpha * y
        
        # Initial Hessian approximation: H_0 = (y^T s / y^T y) * I
        y_last = y_history[-1]
        s_last = s_history[-1]
        gamma = (y_last @ s_last) / (y_last @ y_last + 1e-10)
        
        r = gamma * q
        
        # Second loop (forward)
        for s, y, rho, alpha in zip(s_history, y_history, rho_history, reversed(alpha_values)):
            beta = rho * (y @ r)
            r = r + s * (alpha - beta)
        
        return -r
    
    def _line_search(
        self,
        objective: Callable,
        gradient_fn: Callable,
        params: np.ndarray,
        direction: np.ndarray,
        gradient: np.ndarray,
        c1: float = 1e-4,
    ) -> float:
        """Backtracking line search with Armijo condition.
        
        Args:
            objective: Objective function
            gradient_fn: Gradient function
            params: Current parameters
            direction: Search direction
            gradient: Current gradient
            c1: Armijo condition constant
            
        Returns:
            Step size alpha
        """
        alpha = 1.0
        obj_current = objective(params)
        directional_derivative = float(np.dot(gradient, direction))
        
        for _ in range(self.line_search_max_iter):
            params_new = params + alpha * direction
            obj_new = objective(params_new)
            
            # Armijo condition: sufficient decrease
            if obj_new <= obj_current + c1 * alpha * directional_derivative:
                return alpha
            
            # Reduce step size
            alpha *= 0.5
        
        return alpha
