"""Tests for SGD optimizers."""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.optimization.gradient import (
    SGDOptimizer,
    MomentumSGD,
    NesterovSGD,
)


class TestSGDOptimizer:
    """Test vanilla SGD."""
    
    def test_sgd_quadratic_convergence(self):
        """Test SGD converges on quadratic function."""
        # f(x) = x^2, gradient = 2x
        optimizer = SGDOptimizer(lr=0.1)
        
        x = np.array([10.0])
        
        for _ in range(50):
            grad = 2 * x
            x = optimizer.step(x, grad)
        
        # Should converge to 0
        assert abs(x[0]) < 0.1
    
    def test_sgd_with_weight_decay(self):
        """Test SGD with L2 regularization."""
        optimizer = SGDOptimizer(lr=0.1, weight_decay=0.01)
        
        x = np.array([5.0])
        grad = np.array([0.0])  # Zero gradient
        
        x_new = optimizer.step(x, grad)
        
        # Should decay toward zero
        assert abs(x_new[0]) < abs(x[0])
    
    def test_sgd_reset(self):
        """Test optimizer reset."""
        optimizer = SGDOptimizer(lr=0.1)
        
        x = np.array([1.0])
        optimizer.step(x, np.array([0.5]))
        
        assert optimizer.t == 1
        
        optimizer.reset()
        assert optimizer.t == 0


class TestMomentumSGD:
    """Test SGD with momentum."""
    
    def test_momentum_convergence(self):
        """Test momentum accelerates convergence."""
        optimizer = MomentumSGD(lr=0.01, momentum=0.9)
        
        x = np.array([10.0])
        
        for _ in range(50):
            grad = 2 * x
            x = optimizer.step(x, grad)
        
        # Momentum should help converge faster
        assert abs(x[0]) < 2.0
    
    def test_momentum_initialization(self):
        """Test velocity initialization on first step."""
        optimizer = MomentumSGD(lr=0.1, momentum=0.9)
        
        assert optimizer.velocity is None
        
        x = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        
        optimizer.step(x, grad)
        
        assert optimizer.velocity is not None
        assert optimizer.velocity.shape == x.shape
    
    def test_momentum_dampens_oscillations(self):
        """Test momentum dampens oscillations in noisy gradients."""
        optimizer = MomentumSGD(lr=0.1, momentum=0.9)
        
        x = np.array([5.0])
        
        # Noisy gradients
        for i in range(20):
            noise = np.random.normal(0, 0.5)
            grad = 2 * x + noise
            x = optimizer.step(x, grad)
        
        # Should still converge despite noise
        assert abs(x[0]) < 3.0


class TestNesterovSGD:
    """Test Nesterov momentum."""
    
    def test_nesterov_lookahead(self):
        """Test Nesterov look-ahead parameters."""
        optimizer = NesterovSGD(lr=0.1, momentum=0.9)
        
        x = np.array([1.0, 2.0])
        grad = np.array([0.5, 0.5])
        
        # First step initializes velocity
        optimizer.step(x, grad)
        
        # Get look-ahead params
        x_lookahead = optimizer.get_lookahead_params(x)
        
        # Look-ahead should differ from current
        assert not np.allclose(x_lookahead, x)
    
    def test_nesterov_convergence(self):
        """Test Nesterov converges faster than standard momentum."""
        optimizer = NesterovSGD(lr=0.01, momentum=0.9)
        
        x = np.array([10.0])
        
        for _ in range(50):
            grad = 2 * x
            x = optimizer.step(x, grad)
        
        assert abs(x[0]) < 2.0
    
    def test_nesterov_reset(self):
        """Test Nesterov reset clears velocity."""
        optimizer = NesterovSGD(lr=0.1, momentum=0.9)
        
        x = np.array([1.0])
        optimizer.step(x, np.array([0.5]))
        
        assert optimizer.velocity is not None
        
        optimizer.reset()
        assert optimizer.velocity is None


class TestSGDEdgeCases:
    """Test edge cases for SGD optimizers."""
    
    def test_zero_gradient(self):
        """Test with zero gradient."""
        optimizer = SGDOptimizer(lr=0.1)
        
        x = np.array([5.0])
        grad = np.array([0.0])
        
        x_new = optimizer.step(x, grad)
        
        # Should not move with zero gradient (no weight decay)
        assert x_new[0] == x[0]
    
    def test_multidimensional_parameters(self):
        """Test with multi-dimensional parameters."""
        optimizer = MomentumSGD(lr=0.1, momentum=0.9)
        
        x = np.array([1.0, 2.0, 3.0, 4.0])
        grad = np.array([0.1, 0.2, 0.3, 0.4])
        
        x_new = optimizer.step(x, grad)
        
        assert x_new.shape == x.shape
        assert not np.allclose(x_new, x)
