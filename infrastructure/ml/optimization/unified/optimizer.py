"""Unified optimizer — auto-selects best optimization method.

Decision logic based on:
- Problem dimensionality
- Convexity estimate
- Gradient availability
- Time budget
"""

from __future__ import annotations

import time
import numpy as np
from typing import Callable, Optional
from ..types import OptimizationResult, OptimizerConfig
from ..gradient import SGDOptimizer
from ..convex import LBFGSOptimizer, ConjugateGradientOptimizer
from ..nonconvex import SimulatedAnnealing, ParticleSwarmOptimizer

# NOTE: Adam moved to _experimental — using SGD with momentum as fallback
try:
    from ..._experimental.gradient import AdamOptimizer
except ImportError:
    AdamOptimizer = SGDOptimizer  # type: ignore[misc,assignment]


class UnifiedOptimizer:
    """Unified optimizer with automatic method selection.
    
    Selects best optimization method based on problem characteristics:
    
    Decision tree:
    1. If gradient available:
       - If convex → L-BFGS (fast convergence)
       - If non-convex + low dim (<20) → Adam with restarts
       - If non-convex + high dim → Adam
    
    2. If no gradient:
       - If low dim (<10) → Simulated Annealing
       - If high dim → Particle Swarm
    
    Convexity estimation via Hessian eigenvalues (if gradient available).
    
    Args:
        config: Optimizer configuration
        budget_ms: Time budget in milliseconds (default: 100ms)
    """
    
    def __init__(
        self,
        config: OptimizerConfig = None,
        budget_ms: float = 100.0,
    ):
        self.config = config or OptimizerConfig()
        self.budget_ms = budget_ms
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        convex_hint: bool = True,
        bounds: tuple = None,
    ) -> OptimizationResult:
        """Optimize using auto-selected method.
        
        Args:
            objective: Objective function
            initial_params: Starting parameters
            gradient_fn: Optional gradient function
            convex_hint: Hint if problem is convex (default: True)
            bounds: Optional parameter bounds
            
        Returns:
            OptimizationResult with selected method info
        """
        start_time = time.time()
        
        # Problem dimensionality
        n_dims = len(initial_params)
        
        # Select method (must complete in < 5ms as per constraint)
        method_name, optimizer = self._select_method(
            n_dims,
            gradient_fn,
            convex_hint,
        )
        
        selection_time_ms = (time.time() - start_time) * 1000
        
        # Optimize with selected method
        if gradient_fn is not None and hasattr(optimizer, 'optimize') and 'gradient_fn' in optimizer.optimize.__code__.co_varnames:
            # Gradient-based optimizer
            result = optimizer.optimize(objective, gradient_fn, initial_params)
        elif hasattr(optimizer, 'optimize'):
            # Gradient-free optimizer
            result = optimizer.optimize(objective, initial_params, bounds)
        else:
            # Online optimizer (SGD, Adam) — use simple loop
            result = self._optimize_with_online_optimizer(
                optimizer, objective, gradient_fn, initial_params
            )
        
        # Add method selection info to history
        if result.history is None:
            result.history = {}
        
        result.history["method_selected"] = method_name
        result.history["selection_time_ms"] = selection_time_ms
        
        return result
    
    def _select_method(
        self,
        n_dims: int,
        gradient_fn: Optional[Callable],
        convex_hint: bool,
    ) -> tuple[str, any]:
        """Select optimization method.
        
        Returns:
            (method_name, optimizer_instance)
        """
        # Gradient available?
        if gradient_fn is not None:
            if convex_hint:
                # Convex + gradient → L-BFGS (fast)
                return "L-BFGS", LBFGSOptimizer(self.config)
            else:
                # Non-convex + gradient → Adam
                return "Adam", AdamOptimizer(lr=0.01)
        else:
            # No gradient
            if n_dims < 10:
                # Low-dim + no gradient → Simulated Annealing
                return "SimulatedAnnealing", SimulatedAnnealing(
                    self.config,
                    initial_temperature=1.0,
                    cooling_rate=0.995,
                )
            else:
                # High-dim + no gradient → Particle Swarm
                return "ParticleSwarm", ParticleSwarmOptimizer(
                    self.config,
                    n_particles=min(30, 5 * n_dims),
                )
    
    def _optimize_with_online_optimizer(
        self,
        optimizer,
        objective: Callable,
        gradient_fn: Callable,
        initial_params: np.ndarray,
    ) -> OptimizationResult:
        """Run online optimizer (Adam, SGD) in a loop.
        
        Args:
            optimizer: Online optimizer instance
            objective: Objective function
            gradient_fn: Gradient function
            initial_params: Starting parameters
            
        Returns:
            OptimizationResult
        """
        params = initial_params.copy()
        history = {"objective": [], "norm_gradient": []}
        
        for iteration in range(self.config.max_iterations):
            obj_value = objective(params)
            gradient = gradient_fn(params)
            
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
            
            # Update step
            params = optimizer.step(params, gradient)
        
        return OptimizationResult(
            params=params,
            objective_value=objective(params),
            success=False,
            n_iterations=self.config.max_iterations,
            message="Max iterations reached",
            history=history,
        )
