"""Genetic algorithm for non-convex optimization.

Population-based optimization inspired by natural selection.
"""

from __future__ import annotations

import numpy as np
from typing import Callable
from ..types import OptimizationResult, OptimizerConfig


class GeneticOptimizer:
    """Genetic algorithm optimizer.
    
    Maintains population of candidate solutions.
    Applies selection, crossover, and mutation operators.
    
    Good for non-convex problems with discrete or mixed variables.
    
    Args:
        config: Optimizer configuration
        population_size: Number of individuals
        mutation_rate: Probability of mutation (0-1)
        crossover_rate: Probability of crossover (0-1)
        elite_size: Number of top individuals to preserve
    """
    
    def __init__(
        self,
        config: OptimizerConfig = None,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
    ):
        self.config = config or OptimizerConfig()
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = min(elite_size, population_size)
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        bounds: tuple = None,
    ) -> OptimizationResult:
        """Optimize using genetic algorithm.
        
        Args:
            objective: Objective function to minimize
            initial_params: Reference parameters (for dimensionality)
            bounds: Optional (min, max) bounds
            
        Returns:
            OptimizationResult
        """
        n_dims = len(initial_params)
        
        # Initialize population randomly around initial params
        if bounds is not None:
            population = np.random.uniform(
                bounds[0], bounds[1],
                size=(self.population_size, n_dims)
            )
        else:
            population = initial_params + np.random.normal(
                0, 1.0, size=(self.population_size, n_dims)
            )
        
        history = {"best_objective": [], "mean_objective": []}
        
        for iteration in range(self.config.max_iterations):
            # Evaluate fitness
            fitness = np.array([objective(ind) for ind in population])
            
            # Track best
            best_idx = np.argmin(fitness)
            best_params = population[best_idx].copy()
            best_fitness = fitness[best_idx]
            
            history["best_objective"].append(best_fitness)
            history["mean_objective"].append(np.mean(fitness))
            
            # Check convergence
            if iteration > 50:
                improvement = history["best_objective"][-50] - best_fitness
                if improvement < self.config.tolerance:
                    return OptimizationResult(
                        params=best_params,
                        objective_value=best_fitness,
                        success=True,
                        n_iterations=iteration + 1,
                        message="Converged",
                        history=history,
                    )
            
            # Selection: tournament selection
            selected = self._tournament_selection(population, fitness)
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elite = population[elite_indices].copy()
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutate(child1, bounds)
                child2 = self._mutate(child2, bounds)
                
                offspring.extend([child1, child2])
            
            # New population: elite + offspring
            offspring = np.array(offspring[:self.population_size - self.elite_size])
            population = np.vstack([elite, offspring])
        
        # Final evaluation
        fitness = np.array([objective(ind) for ind in population])
        best_idx = np.argmin(fitness)
        
        return OptimizationResult(
            params=population[best_idx],
            objective_value=fitness[best_idx],
            success=True,
            n_iterations=self.config.max_iterations,
            message="Max iterations reached",
            history=history,
        )
    
    def _tournament_selection(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        tournament_size: int = 3,
    ) -> np.ndarray:
        """Select individuals via tournament selection."""
        selected = []
        
        for _ in range(len(population)):
            # Random tournament
            indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness[indices]
            winner_idx = indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return np.array(selected)
    
    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        n = len(parent1)
        point = np.random.randint(1, n)
        
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        return child1, child2
    
    def _mutate(
        self,
        individual: np.ndarray,
        bounds: tuple = None,
    ) -> np.ndarray:
        """Gaussian mutation."""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] += np.random.normal(0, 0.1)
        
        # Apply bounds
        if bounds is not None:
            mutated = np.clip(mutated, bounds[0], bounds[1])
        
        return mutated
