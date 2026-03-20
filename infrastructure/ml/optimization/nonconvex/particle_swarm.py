"""Particle swarm optimization.

Swarm intelligence for non-convex global optimization.
"""

from __future__ import annotations

import numpy as np
from typing import Callable
from ..types import OptimizationResult, OptimizerConfig


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization (PSO).
    
    Population of particles explore search space.
    Each particle has position and velocity, influenced by:
    - Personal best position
    - Global best position
    
    Update rules:
        v_i = w*v_i + c1*r1*(p_best - x_i) + c2*r2*(g_best - x_i)
        x_i = x_i + v_i
    
    Good for continuous non-convex problems.
    
    Args:
        config: Optimizer configuration
        n_particles: Number of particles
        w: Inertia weight (typically 0.7-0.9)
        c1: Cognitive coefficient (personal best attraction)
        c2: Social coefficient (global best attraction)
    """
    
    def __init__(
        self,
        config: OptimizerConfig = None,
        n_particles: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        self.config = config or OptimizerConfig()
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        bounds: tuple = None,
    ) -> OptimizationResult:
        """Optimize using particle swarm.
        
        Args:
            objective: Objective function to minimize
            initial_params: Reference parameters
            bounds: Optional (min, max) bounds
            
        Returns:
            OptimizationResult
        """
        n_dims = len(initial_params)
        
        # Initialize particles
        if bounds is not None:
            positions = np.random.uniform(
                bounds[0], bounds[1],
                size=(self.n_particles, n_dims)
            )
        else:
            positions = initial_params + np.random.normal(
                0, 1.0, size=(self.n_particles, n_dims)
            )
        
        # Initialize velocities
        velocities = np.random.normal(0, 0.1, size=(self.n_particles, n_dims))
        
        # Evaluate initial positions
        fitness = np.array([objective(pos) for pos in positions])
        
        # Personal best
        p_best_positions = positions.copy()
        p_best_fitness = fitness.copy()
        
        # Global best
        g_best_idx = np.argmin(fitness)
        g_best_position = positions[g_best_idx].copy()
        g_best_fitness = fitness[g_best_idx]
        
        history = {"best_objective": [], "mean_objective": []}
        
        for iteration in range(self.config.max_iterations):
            history["best_objective"].append(g_best_fitness)
            history["mean_objective"].append(np.mean(fitness))
            
            # Check convergence
            if iteration > 50:
                improvement = history["best_objective"][-50] - g_best_fitness
                if improvement < self.config.tolerance:
                    return OptimizationResult(
                        params=g_best_position,
                        objective_value=g_best_fitness,
                        success=True,
                        n_iterations=iteration + 1,
                        message="Converged",
                        history=history,
                    )
            
            # Update particles
            for i in range(self.n_particles):
                # Random factors
                r1 = np.random.random(n_dims)
                r2 = np.random.random(n_dims)
                
                # Velocity update
                cognitive = self.c1 * r1 * (p_best_positions[i] - positions[i])
                social = self.c2 * r2 * (g_best_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Position update
                positions[i] = positions[i] + velocities[i]
                
                # Apply bounds
                if bounds is not None:
                    positions[i] = np.clip(positions[i], bounds[0], bounds[1])
                
                # Evaluate
                fitness[i] = objective(positions[i])
                
                # Update personal best
                if fitness[i] < p_best_fitness[i]:
                    p_best_positions[i] = positions[i].copy()
                    p_best_fitness[i] = fitness[i]
                
                # Update global best
                if fitness[i] < g_best_fitness:
                    g_best_position = positions[i].copy()
                    g_best_fitness = fitness[i]
        
        return OptimizationResult(
            params=g_best_position,
            objective_value=g_best_fitness,
            success=True,
            n_iterations=self.config.max_iterations,
            message="Max iterations reached",
            history=history,
        )
