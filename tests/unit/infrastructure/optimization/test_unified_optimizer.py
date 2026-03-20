"""Tests for unified optimizer."""

import pytest
import numpy as np
import time

from iot_machine_learning.infrastructure.ml.optimization.unified import UnifiedOptimizer
from iot_machine_learning.infrastructure.ml.optimization.types import OptimizerConfig


class TestUnifiedOptimizerSelection:
    """Test method selection logic."""
    
    def test_selects_lbfgs_for_convex_with_gradient(self):
        """Test selects L-BFGS for convex problem with gradient."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        def gradient(x):
            return 2 * x
        
        config = OptimizerConfig(max_iterations=50)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.array([5.0, 3.0]),
            gradient_fn=gradient,
            convex_hint=True,
        )
        
        assert result.history["method_selected"] == "L-BFGS"
        assert result.success or np.linalg.norm(result.params) < 1.0
    
    def test_selects_adam_for_nonconvex_with_gradient(self):
        """Test selects Adam for non-convex problem with gradient."""
        def objective(x):
            return float(x[0]**4)
        
        def gradient(x):
            return 4 * x**3
        
        config = OptimizerConfig(max_iterations=100)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.array([2.0]),
            gradient_fn=gradient,
            convex_hint=False,
        )
        
        assert result.history["method_selected"] == "Adam"
    
    def test_selects_sa_for_lowdim_no_gradient(self):
        """Test selects SA for low-dim problem without gradient."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        config = OptimizerConfig(max_iterations=200)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.array([5.0, 3.0]),  # 2D
            gradient_fn=None,
            convex_hint=False,
        )
        
        assert result.history["method_selected"] == "SimulatedAnnealing"
    
    def test_selects_pso_for_highdim_no_gradient(self):
        """Test selects PSO for high-dim problem without gradient."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        config = OptimizerConfig(max_iterations=100)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.random.randn(15),  # 15D
            gradient_fn=None,
            convex_hint=False,
        )
        
        assert result.history["method_selected"] == "ParticleSwarm"
    
    def test_selection_time_under_5ms(self):
        """Test method selection completes in < 5ms."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        def gradient(x):
            return 2 * x
        
        config = OptimizerConfig(max_iterations=10)
        optimizer = UnifiedOptimizer(config)
        
        start = time.time()
        result = optimizer.optimize(
            objective,
            np.array([1.0]),
            gradient_fn=gradient,
        )
        selection_time = result.history["selection_time_ms"]
        
        # Selection should be fast (< 5ms as per constraint)
        assert selection_time < 5.0


class TestUnifiedOptimizerConvergence:
    """Test unified optimizer achieves good solutions."""
    
    def test_unified_quadratic_convergence(self):
        """Test unified optimizer converges on quadratic."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        def gradient(x):
            return 2 * x
        
        config = OptimizerConfig(max_iterations=50, tolerance=1e-4)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.array([10.0, -5.0]),
            gradient_fn=gradient,
        )
        
        assert np.linalg.norm(result.params) < 0.5
    
    def test_unified_nonconvex_finds_good_solution(self):
        """Test unified optimizer on non-convex function."""
        def objective(x):
            return float(x[0]**4 - 2*x[0]**2)
        
        def gradient(x):
            return 4*x**3 - 4*x
        
        config = OptimizerConfig(max_iterations=100)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.array([2.0]),
            gradient_fn=gradient,
            convex_hint=False,
        )
        
        # Should find a good local minimum
        assert result.objective_value < 0.0
    
    def test_unified_no_gradient_still_works(self):
        """Test unified optimizer works without gradient."""
        def objective(x):
            return float(np.sum((x - 3.0) ** 2))
        
        config = OptimizerConfig(max_iterations=200)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.array([0.0, 0.0]),
            gradient_fn=None,
        )
        
        # Should get close to [3, 3]
        assert np.linalg.norm(result.params - 3.0) < 2.0


class TestUnifiedOptimizerWithBounds:
    """Test unified optimizer with parameter bounds."""
    
    def test_unified_respects_bounds_sa(self):
        """Test bounds are respected when SA is selected."""
        def objective(x):
            return float(x[0] ** 2)
        
        config = OptimizerConfig(max_iterations=100)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.array([0.5]),
            gradient_fn=None,  # Will select SA (low dim)
            bounds=(-1.0, 1.0),
        )
        
        assert -1.0 <= result.params[0] <= 1.0
    
    def test_unified_respects_bounds_pso(self):
        """Test bounds are respected when PSO is selected."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        config = OptimizerConfig(max_iterations=50)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.random.randn(12),  # High dim → PSO
            gradient_fn=None,
            bounds=(-2.0, 2.0),
        )
        
        assert np.all(result.params >= -2.0)
        assert np.all(result.params <= 2.0)


class TestUnifiedOptimizerEdgeCases:
    """Test edge cases for unified optimizer."""
    
    def test_unified_single_parameter(self):
        """Test with single parameter."""
        def objective(x):
            return float(x[0] ** 2)
        
        def gradient(x):
            return 2 * x
        
        config = OptimizerConfig(max_iterations=20)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.array([5.0]),
            gradient_fn=gradient,
        )
        
        assert result.params.shape == (1,)
        assert abs(result.params[0]) < 1.0
    
    def test_unified_many_parameters(self):
        """Test with many parameters."""
        n = 50
        
        def objective(x):
            return float(np.sum(x ** 2))
        
        def gradient(x):
            return 2 * x
        
        config = OptimizerConfig(max_iterations=100)
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.random.randn(n),
            gradient_fn=gradient,
        )
        
        assert result.params.shape == (n,)
        assert np.linalg.norm(result.params) < 5.0
    
    def test_unified_graceful_fail(self):
        """Test unified optimizer returns result even if not converged."""
        def objective(x):
            return float(np.sum(x ** 2))
        
        config = OptimizerConfig(max_iterations=5, tolerance=1e-10)  # Very strict
        optimizer = UnifiedOptimizer(config)
        
        result = optimizer.optimize(
            objective,
            np.array([10.0]),
            gradient_fn=lambda x: 2*x,
        )
        
        # Should return a result even if not converged
        assert result.params is not None
        assert result.objective_value is not None
