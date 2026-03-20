"""Simulated annealing for non-convex optimization.

Escapes local minima via probabilistic acceptance of worse solutions.
"""

from __future__ import annotations

import numpy as np
from typing import Callable
from ..types import OptimizationResult, OptimizerConfig


class SimulatedAnnealing:
    """Simulated annealing optimizer.
    
    Probabilistic optimization inspired by metallurgy annealing process.
    Accepts worse solutions with probability P = exp(-ΔE / T).
    
    Temperature T decreases over time (cooling schedule), reducing
    probability of accepting worse solutions.
    
    Effective for non-convex landscapes with many local minima.
    
    Args:
        config: Optimizer configuration
        initial_temperature: Starting temperature
        cooling_rate: Temperature decay (typically 0.95-0.999)
        step_size: Perturbation size for candidate generation
    """
    
    def __init__(
        self,
        config: OptimizerConfig = None,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.995,
        step_size: float = 0.1,
    ):
        self.config = config or OptimizerConfig()
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.step_size = step_size
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        bounds: tuple = None,
    ) -> OptimizationResult:
        """Optimize using simulated annealing.
        
        Args:
            objective: Objective function to minimize
            initial_params: Starting parameters
            bounds: Optional (min, max) bounds for parameters
            
        Returns:
            OptimizationResult
        """
        # Current solution
        params_current = initial_params.copy()
        energy_current = objective(params_current)
        
        # Best solution found
        params_best = params_current.copy()
        energy_best = energy_current
        
        # Temperature
        temperature = self.initial_temperature
        
        history = {
            "objective": [energy_current],
            "temperature": [temperature],
            "acceptance_rate": [],
        }
        
        n_accepted = 0
        n_total = 0
        
        for iteration in range(self.config.max_iterations):
            # Generate candidate (random perturbation)
            perturbation = np.random.normal(0, self.step_size, size=params_current.shape)
            params_candidate = params_current + perturbation
            
            # Apply bounds if specified
            if bounds is not None:
                params_candidate = np.clip(params_candidate, bounds[0], bounds[1])
            
            energy_candidate = objective(params_candidate)
            
            # Compute energy change
            delta_energy = energy_candidate - energy_current
            
            # Acceptance probability
            if delta_energy < 0:
                # Better solution → always accept
                accept = True
            else:
                # Worse solution → accept with probability exp(-ΔE / T)
                acceptance_prob = np.exp(-delta_energy / (temperature + 1e-10))
                accept = np.random.random() < acceptance_prob
            
            # Update if accepted
            if accept:
                params_current = params_candidate
                energy_current = energy_candidate
                n_accepted += 1
                
                # Update best
                if energy_current < energy_best:
                    params_best = params_current.copy()
                    energy_best = energy_current
            
            n_total += 1
            
            # Cool down temperature
            temperature *= self.cooling_rate
            
            # Record history
            history["objective"].append(energy_best)
            history["temperature"].append(temperature)
            if n_total > 0:
                history["acceptance_rate"].append(n_accepted / n_total)
            
            # Early stopping check
            if self.config.early_stopping and iteration > 100:
                recent_improvement = history["objective"][-100] - energy_best
                if recent_improvement < self.config.tolerance:
                    return OptimizationResult(
                        params=params_best,
                        objective_value=energy_best,
                        success=True,
                        n_iterations=iteration + 1,
                        message="Converged (no recent improvement)",
                        history=history,
                    )
        
        # Check if we found a good solution
        success = energy_best < energy_current or n_accepted > 0
        
        return OptimizationResult(
            params=params_best,
            objective_value=energy_best,
            success=success,
            n_iterations=self.config.max_iterations,
            message=f"Completed (acceptance rate: {n_accepted/n_total:.2%})",
            history=history,
        )
