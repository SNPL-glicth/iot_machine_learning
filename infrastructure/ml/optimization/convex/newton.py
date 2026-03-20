"""Newton-Raphson and quasi-Newton optimization.

Newton methods use second-order information (Hessian) for faster convergence.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional
from ..types import OptimizationResult, OptimizerConfig


class NewtonRaphsonOptimizer:
    """Newton-Raphson optimizer.
    
    Uses second-order Taylor expansion:
        θ_{t+1} = θ_t - H^{-1} * ∇f(θ_t)
    
    where H = Hessian matrix
    
    Quadratic convergence near minimum, but requires Hessian computation.
    
    Args:
        config: Optimizer configuration
        damping: Damping factor for stability (adds λI to Hessian)
    """
    
    def __init__(
        self,
        config: OptimizerConfig = None,
        damping: float = 1e-4,
    ):
        self.config = config or OptimizerConfig()
        self.damping = damping
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        hessian_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
    ) -> OptimizationResult:
        """Optimize using Newton-Raphson.
        
        Args:
            objective: Objective function f(θ)
            gradient_fn: Gradient function ∇f(θ)
            hessian_fn: Hessian function H(θ)
            initial_params: Starting parameters
            
        Returns:
            OptimizationResult
        """
        params = initial_params.copy()
        history = {"objective": [], "norm_gradient": []}
        
        for iteration in range(self.config.max_iterations):
            # Evaluate objective and derivatives
            obj_value = objective(params)
            gradient = gradient_fn(params)
            hessian = hessian_fn(params)
            
            # Record history
            history["objective"].append(obj_value)
            history["norm_gradient"].append(float(np.linalg.norm(gradient)))
            
            # Check convergence
            if np.linalg.norm(gradient) < self.config.tolerance:
                return OptimizationResult(
                    params=params,
                    objective_value=obj_value,
                    success=True,
                    n_iterations=iteration + 1,
                    message="Converged (gradient norm < tolerance)",
                    history=history,
                )
            
            # Damped Hessian: H_damped = H + λI
            damped_hessian = hessian + self.damping * np.eye(len(params))
            
            try:
                # Newton step: Δθ = -H^{-1} * ∇f
                delta = -np.linalg.solve(damped_hessian, gradient)
            except np.linalg.LinAlgError:
                return OptimizationResult(
                    params=params,
                    objective_value=obj_value,
                    success=False,
                    n_iterations=iteration + 1,
                    message="Singular Hessian matrix",
                    history=history,
                )
            
            # Update parameters
            params = params + delta
        
        # Max iterations reached
        return OptimizationResult(
            params=params,
            objective_value=objective(params),
            success=False,
            n_iterations=self.config.max_iterations,
            message="Max iterations reached",
            history=history,
        )


class QuasiNewtonOptimizer:
    """Quasi-Newton optimizer (BFGS approximation).
    
    Approximates Hessian inverse using gradient history:
        B_{k+1} = B_k + correction terms
    
    Avoids expensive Hessian computation.
    
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
        """Optimize using BFGS.
        
        Args:
            objective: Objective function
            gradient_fn: Gradient function
            initial_params: Starting parameters
            
        Returns:
            OptimizationResult
        """
        params = initial_params.copy()
        n = len(params)
        
        # Initialize inverse Hessian approximation as identity
        B_inv = np.eye(n)
        
        gradient = gradient_fn(params)
        history = {"objective": [], "norm_gradient": []}
        
        for iteration in range(self.config.max_iterations):
            obj_value = objective(params)
            history["objective"].append(obj_value)
            history["norm_gradient"].append(float(np.linalg.norm(gradient)))
            
            # Check convergence
            if np.linalg.norm(gradient) < self.config.tolerance:
                return OptimizationResult(
                    params=params,
                    objective_value=obj_value,
                    success=True,
                    n_iterations=iteration + 1,
                    message="Converged",
                    history=history,
                )
            
            # Search direction: d = -B^{-1} * g
            direction = -B_inv @ gradient
            
            # Line search (simple backtracking)
            alpha = 1.0
            params_new = params + alpha * direction
            gradient_new = gradient_fn(params_new)
            
            # BFGS update
            s = params_new - params  # Step
            y = gradient_new - gradient  # Gradient change
            
            # Update inverse Hessian approximation
            rho = 1.0 / (y @ s + 1e-10)
            
            if rho > 0:  # Ensure positive definite
                I = np.eye(n)
                B_inv = (I - rho * np.outer(s, y)) @ B_inv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            
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
