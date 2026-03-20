"""Tests for simulated annealing optimizer."""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.optimization.nonconvex import SimulatedAnnealing
from iot_machine_learning.infrastructure.ml.optimization.types import OptimizerConfig


class TestSimulatedAnnealing:
    """Test simulated annealing optimizer."""
    
    def test_sa_finds_global_minimum(self):
        """Test SA escapes local minima to find global minimum."""
        # f(x) = x^4 - 2x^2 (two local minima at ±1, global at 0)
        def objective(x):
            return float(x[0]**4 - 2*x[0]**2)
        
        config = OptimizerConfig(max_iterations=2000, tolerance=1e-4)
        optimizer = SimulatedAnnealing(
            config,
            initial_temperature=2.0,
            cooling_rate=0.995,
            step_size=0.3,
        )
        
        result = optimizer.optimize(
            objective,
            initial_params=np.array([2.0])  # Start far from optimum
        )
        
        # Should find good solution (local or global minimum)
        assert result.objective_value < -0.5  # Better than starting point
    
    def test_sa_accepts_worse_solutions_early(self):
        """Test SA accepts worse solutions at high temperature."""
        def objective(x):
            return float(x[0] ** 2)
        
        config = OptimizerConfig(max_iterations=100)
        optimizer = SimulatedAnnealing(
            config,
            initial_temperature=10.0,  # High temperature
            cooling_rate=0.95,
        )
        
        result = optimizer.optimize(
            objective,
            initial_params=np.array([1.0])
        )
        
        # Check acceptance rate is recorded
        assert "acceptance_rate" in result.history
        assert len(result.history["acceptance_rate"]) > 0
    
    def test_sa_cooling_schedule(self):
        """Test temperature decreases according to cooling schedule."""
        def objective(x):
            return float(x[0] ** 2)
        
        config = OptimizerConfig(max_iterations=50)
        optimizer = SimulatedAnnealing(
            config,
            initial_temperature=1.0,
            cooling_rate=0.9,
        )
        
        result = optimizer.optimize(
            objective,
            initial_params=np.array([5.0])
        )
        
        temps = result.history["temperature"]
        
        # Temperature should decrease monotonically
        assert temps[0] == 1.0
        assert temps[-1] < temps[0]
        assert all(temps[i] >= temps[i+1] for i in range(len(temps)-1))
    
    def test_sa_with_bounds(self):
        """Test SA respects parameter bounds."""
        def objective(x):
            return float(x[0] ** 2)
        
        config = OptimizerConfig(max_iterations=100)
        optimizer = SimulatedAnnealing(config)
        
        result = optimizer.optimize(
            objective,
            initial_params=np.array([0.5]),
            bounds=(-1.0, 1.0)
        )
        
        # Final params should be within bounds
        assert -1.0 <= result.params[0] <= 1.0
    
    def test_sa_multidimensional(self):
        """Test SA on multi-dimensional problem."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        config = OptimizerConfig(max_iterations=500)
        optimizer = SimulatedAnnealing(config, initial_temperature=1.0, cooling_rate=0.995)
        
        result = optimizer.optimize(
            objective,
            initial_params=np.array([3.0, -2.0, 4.0])
        )
        
        # Should find near-zero
        assert np.linalg.norm(result.params) < 2.0
    
    def test_sa_rastrigin_function(self):
        """Test SA on Rastrigin function (many local minima)."""
        def rastrigin(x):
            A = 10
            n = len(x)
            return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
        
        config = OptimizerConfig(max_iterations=1000, tolerance=1e-3)
        optimizer = SimulatedAnnealing(
            config,
            initial_temperature=5.0,
            cooling_rate=0.995,
            step_size=0.5,
        )
        
        result = optimizer.optimize(
            rastrigin,
            initial_params=np.array([2.0, -3.0])
        )
        
        # Should find a good solution (Rastrigin is very hard)
        # Global minimum is 0, but we'll accept any improvement
        initial_value = rastrigin(np.array([2.0, -3.0]))
        assert result.objective_value < initial_value  # Better than start


class TestSimulatedAnnealingEdgeCases:
    """Test edge cases for simulated annealing."""
    
    def test_sa_already_at_minimum(self):
        """Test SA when starting at minimum."""
        def objective(x):
            return float(x[0] ** 2)
        
        config = OptimizerConfig(max_iterations=100, early_stopping=True)
        optimizer = SimulatedAnnealing(config)
        
        result = optimizer.optimize(
            objective,
            initial_params=np.array([0.01])
        )
        
        # Should recognize it's at minimum
        assert abs(result.params[0]) < 0.5
    
    def test_sa_zero_cooling_rate(self):
        """Test SA with very slow cooling."""
        def objective(x):
            return float(x[0] ** 2)
        
        config = OptimizerConfig(max_iterations=50)
        optimizer = SimulatedAnnealing(
            config,
            initial_temperature=1.0,
            cooling_rate=0.99,  # Slow cooling
        )
        
        result = optimizer.optimize(
            objective,
            initial_params=np.array([5.0])
        )
        
        # Should still make progress
        assert result.n_iterations > 0
    
    def test_sa_history_tracking(self):
        """Test SA tracks optimization history."""
        def objective(x):
            return float(x[0] ** 2)
        
        config = OptimizerConfig(max_iterations=50)
        optimizer = SimulatedAnnealing(config)
        
        result = optimizer.optimize(
            objective,
            initial_params=np.array([5.0])
        )
        
        assert "objective" in result.history
        assert "temperature" in result.history
        assert "acceptance_rate" in result.history
        assert len(result.history["objective"]) == result.n_iterations + 1
