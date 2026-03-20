"""Tests for Adam and adaptive optimizers."""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.optimization.gradient import (
    AdamOptimizer,
    AdaGradOptimizer,
    RMSPropOptimizer,
)


class TestAdamOptimizer:
    """Test Adam optimizer."""
    
    def test_adam_convergence(self):
        """Test Adam converges on simple function."""
        optimizer = AdamOptimizer(lr=0.1)
        
        x = np.array([10.0, -5.0])
        
        for _ in range(200):
            grad = 2 * x  # Quadratic gradient
            x = optimizer.step(x, grad)
        
        # Adam is adaptive, may take longer than momentum
        assert np.linalg.norm(x) < 2.0
    
    def test_adam_bias_correction(self):
        """Test bias correction in early iterations."""
        optimizer = AdamOptimizer(lr=0.01, beta1=0.9, beta2=0.999)
        
        x = np.array([1.0])
        grad = np.array([1.0])
        
        # First step
        x1 = optimizer.step(x, grad)
        
        # Bias correction should prevent large first step
        assert abs(x1[0] - x[0]) < 0.5
    
    def test_adam_moment_initialization(self):
        """Test first and second moment initialization."""
        optimizer = AdamOptimizer()
        
        assert optimizer.m is None
        assert optimizer.v is None
        
        x = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        
        optimizer.step(x, grad)
        
        assert optimizer.m is not None
        assert optimizer.v is not None
        assert optimizer.m.shape == x.shape
        assert optimizer.v.shape == x.shape
    
    def test_adam_with_weight_decay(self):
        """Test Adam with L2 regularization."""
        optimizer = AdamOptimizer(lr=0.1, weight_decay=0.01)
        
        x = np.array([5.0])
        grad = np.array([0.0])
        
        x_new = optimizer.step(x, grad)
        
        # Weight decay should pull toward zero
        assert abs(x_new[0]) < abs(x[0])
    
    def test_adam_reset(self):
        """Test Adam reset clears moments."""
        optimizer = AdamOptimizer()
        
        x = np.array([1.0])
        optimizer.step(x, np.array([0.5]))
        
        assert optimizer.m is not None
        assert optimizer.t == 1
        
        optimizer.reset()
        
        assert optimizer.m is None
        assert optimizer.v is None
        assert optimizer.t == 0


class TestAdaGradOptimizer:
    """Test AdaGrad optimizer."""
    
    def test_adagrad_convergence(self):
        """Test AdaGrad converges."""
        optimizer = AdaGradOptimizer(lr=1.0)
        
        x = np.array([10.0])
        
        for _ in range(100):
            grad = 2 * x
            x = optimizer.step(x, grad)
        
        assert abs(x[0]) < 1.0
    
    def test_adagrad_adaptive_lr(self):
        """Test AdaGrad adapts learning rate per parameter."""
        optimizer = AdaGradOptimizer(lr=0.1)
        
        x = np.array([1.0, 1.0])
        
        # One dimension gets larger gradients
        for _ in range(10):
            grad = np.array([1.0, 0.1])
            x = optimizer.step(x, grad)
        
        # Dimension with larger gradients should have smaller effective LR
        assert optimizer.G[0] > optimizer.G[1]
    
    def test_adagrad_initialization(self):
        """Test accumulator initialization."""
        optimizer = AdaGradOptimizer()
        
        assert optimizer.G is None
        
        x = np.array([1.0, 2.0])
        optimizer.step(x, np.array([0.1, 0.2]))
        
        assert optimizer.G is not None


class TestRMSPropOptimizer:
    """Test RMSProp optimizer."""
    
    def test_rmsprop_convergence(self):
        """Test RMSProp converges."""
        optimizer = RMSPropOptimizer(lr=0.1, rho=0.9)
        
        x = np.array([10.0])
        
        for _ in range(100):
            grad = 2 * x
            x = optimizer.step(x, grad)
        
        assert abs(x[0]) < 2.0
    
    def test_rmsprop_moving_average(self):
        """Test RMSProp uses moving average."""
        optimizer = RMSPropOptimizer(lr=0.1, rho=0.9)
        
        x = np.array([1.0])
        
        # First gradient
        optimizer.step(x, np.array([1.0]))
        E_g2_first = optimizer.E_g2.copy()
        
        # Second gradient
        optimizer.step(x, np.array([2.0]))
        E_g2_second = optimizer.E_g2.copy()
        
        # Moving average should change
        assert E_g2_second[0] > E_g2_first[0]
    
    def test_rmsprop_fixes_adagrad_decay(self):
        """Test RMSProp doesn't suffer from AdaGrad's aggressive decay."""
        rmsprop = RMSPropOptimizer(lr=0.1, rho=0.9)
        
        x_rmsprop = np.array([10.0])
        
        # Many iterations
        for _ in range(200):
            grad = 2 * x_rmsprop
            x_rmsprop = rmsprop.step(x_rmsprop, grad)
        
        # Should still make progress (not stuck)
        assert abs(x_rmsprop[0]) < 5.0


class TestAdaptiveOptimizerEdgeCases:
    """Test edge cases for adaptive optimizers."""
    
    def test_zero_gradient(self):
        """Test adaptive optimizers with zero gradient."""
        adam = AdamOptimizer(lr=0.1)
        
        x = np.array([5.0])
        grad = np.array([0.0])
        
        x_new = adam.step(x, grad)
        
        # Should not crash
        assert x_new.shape == x.shape
    
    def test_very_small_gradients(self):
        """Test with very small gradients (numerical stability)."""
        adam = AdamOptimizer(lr=0.1, epsilon=1e-8)
        
        x = np.array([1.0])
        grad = np.array([1e-10])
        
        x_new = adam.step(x, grad)
        
        # Should handle gracefully
        assert np.isfinite(x_new[0])
    
    def test_large_gradients(self):
        """Test with large gradients."""
        adam = AdamOptimizer(lr=0.001)
        
        x = np.array([1.0])
        grad = np.array([1000.0])
        
        x_new = adam.step(x, grad)
        
        # Adaptive LR should prevent explosion
        assert np.isfinite(x_new[0])
