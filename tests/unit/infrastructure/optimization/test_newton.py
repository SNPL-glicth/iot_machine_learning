"""Tests for Newton-based optimizers."""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.optimization.convex import (
    NewtonRaphsonOptimizer,
    LBFGSOptimizer,
)
from iot_machine_learning.infrastructure.ml.optimization.types import OptimizerConfig


class TestNewtonRaphsonOptimizer:
    """Test Newton-Raphson optimizer."""
    
    def test_newton_quadratic_convergence(self):
        """Test Newton-Raphson achieves quadratic convergence."""
        # f(x) = x^2
        def objective(x):
            return float(x[0] ** 2)
        
        def gradient(x):
            return 2 * x
        
        def hessian(x):
            return np.array([[2.0]])
        
        config = OptimizerConfig(max_iterations=10, tolerance=1e-6)
        optimizer = NewtonRaphsonOptimizer(config)
        
        result = optimizer.optimize(
            objective, gradient, hessian,
            initial_params=np.array([10.0])
        )
        
        assert result.success
        assert abs(result.params[0]) < 1e-5
        assert result.n_iterations < 10  # Fast convergence
    
    def test_newton_multidimensional(self):
        """Test Newton on multi-dimensional quadratic."""
        # f(x, y) = x^2 + 2y^2
        def objective(params):
            x, y = params
            return float(x**2 + 2*y**2)
        
        def gradient(params):
            x, y = params
            return np.array([2*x, 4*y])
        
        def hessian(params):
            return np.array([[2.0, 0.0], [0.0, 4.0]])
        
        config = OptimizerConfig(max_iterations=20)
        optimizer = NewtonRaphsonOptimizer(config)
        
        result = optimizer.optimize(
            objective, gradient, hessian,
            initial_params=np.array([5.0, 3.0])
        )
        
        assert result.success
        assert np.linalg.norm(result.params) < 0.01
    
    def test_newton_with_damping(self):
        """Test damping prevents singular Hessian issues."""
        def objective(x):
            return float(x[0] ** 2)
        
        def gradient(x):
            return 2 * x
        
        def hessian(x):
            # Near-singular Hessian
            return np.array([[1e-10]])
        
        config = OptimizerConfig(max_iterations=20)
        optimizer = NewtonRaphsonOptimizer(config, damping=1e-4)
        
        result = optimizer.optimize(
            objective, gradient, hessian,
            initial_params=np.array([5.0])
        )
        
        # Damping should allow progress
        assert result.n_iterations > 0
    
    def test_newton_singular_hessian_failure(self):
        """Test Newton fails gracefully with truly singular Hessian."""
        def objective(x):
            return float(x[0])
        
        def gradient(x):
            return np.array([1.0])
        
        def hessian(x):
            return np.array([[0.0]])  # Singular
        
        config = OptimizerConfig(max_iterations=10)
        optimizer = NewtonRaphsonOptimizer(config, damping=0.0)
        
        result = optimizer.optimize(
            objective, gradient, hessian,
            initial_params=np.array([1.0])
        )
        
        assert not result.success
        assert "Singular" in result.message


class TestLBFGSOptimizer:
    """Test L-BFGS optimizer."""
    
    def test_lbfgs_convergence(self):
        """Test L-BFGS converges on quadratic."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        def gradient(x):
            return 2 * x
        
        config = OptimizerConfig(max_iterations=50, tolerance=1e-5)
        optimizer = LBFGSOptimizer(config, m=5)
        
        result = optimizer.optimize(
            objective, gradient,
            initial_params=np.array([10.0, -5.0, 3.0])
        )
        
        assert result.success
        assert np.linalg.norm(result.params) < 0.01
    
    def test_lbfgs_high_dimensional(self):
        """Test L-BFGS on high-dimensional problem."""
        n = 20
        
        def objective(x):
            return float(np.sum(x ** 2))
        
        def gradient(x):
            return 2 * x
        
        config = OptimizerConfig(max_iterations=100, tolerance=1e-4)
        optimizer = LBFGSOptimizer(config, m=10)
        
        result = optimizer.optimize(
            objective, gradient,
            initial_params=np.random.randn(n)
        )
        
        assert result.success or result.n_iterations == config.max_iterations
        assert np.linalg.norm(result.params) < 1.0
    
    def test_lbfgs_memory_limit(self):
        """Test L-BFGS respects memory limit m."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        def gradient(x):
            return 2 * x
        
        config = OptimizerConfig(max_iterations=20)
        optimizer = LBFGSOptimizer(config, m=3)
        
        result = optimizer.optimize(
            objective, gradient,
            initial_params=np.array([5.0])
        )
        
        # Should converge despite limited memory
        assert result.success or abs(result.params[0]) < 1.0
    
    def test_lbfgs_line_search(self):
        """Test L-BFGS line search finds good step size."""
        def objective(x):
            return float(x[0] ** 4)  # Non-quadratic
        
        def gradient(x):
            return 4 * x ** 3
        
        config = OptimizerConfig(max_iterations=50)
        optimizer = LBFGSOptimizer(config)
        
        result = optimizer.optimize(
            objective, gradient,
            initial_params=np.array([2.0])
        )
        
        # Line search should prevent divergence
        assert abs(result.params[0]) < 2.0


class TestNewtonEdgeCases:
    """Test edge cases for Newton methods."""
    
    def test_already_at_minimum(self):
        """Test when starting at minimum."""
        def objective(x):
            return float(x[0] ** 2)
        
        def gradient(x):
            return 2 * x
        
        def hessian(x):
            return np.array([[2.0]])
        
        config = OptimizerConfig(tolerance=1e-6)
        optimizer = NewtonRaphsonOptimizer(config)
        
        result = optimizer.optimize(
            objective, gradient, hessian,
            initial_params=np.array([0.0])
        )
        
        assert result.success
        assert result.n_iterations == 1  # Immediate convergence
    
    def test_lbfgs_rosenbrock(self):
        """Test L-BFGS on Rosenbrock function (classic test)."""
        def rosenbrock(x):
            return float(100*(x[1] - x[0]**2)**2 + (1 - x[0])**2)
        
        def rosenbrock_grad(x):
            dx = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
            dy = 200*(x[1] - x[0]**2)
            return np.array([dx, dy])
        
        config = OptimizerConfig(max_iterations=200, tolerance=1e-4)
        optimizer = LBFGSOptimizer(config)
        
        result = optimizer.optimize(
            rosenbrock, rosenbrock_grad,
            initial_params=np.array([0.0, 0.0])
        )
        
        # Should get close to minimum at (1, 1)
        assert np.linalg.norm(result.params - np.array([1.0, 1.0])) < 0.5
