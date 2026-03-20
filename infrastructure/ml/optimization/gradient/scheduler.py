"""Learning rate schedulers.

Adjust learning rate during training for better convergence.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class LearningRateScheduler(ABC):
    """Base class for learning rate schedulers."""
    
    @abstractmethod
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for given epoch.
        
        Args:
            epoch: Current epoch/iteration
            
        Returns:
            Learning rate
        """
        pass


class StepLRScheduler(LearningRateScheduler):
    """Step decay learning rate scheduler.
    
    Reduces learning rate by factor every step_size epochs:
        lr = lr_0 * gamma^(epoch // step_size)
    
    Args:
        initial_lr: Initial learning rate
        step_size: Period of learning rate decay
        gamma: Multiplicative factor (default: 0.1)
    """
    
    def __init__(
        self,
        initial_lr: float,
        step_size: int,
        gamma: float = 0.1,
    ):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for epoch."""
        decay_factor = self.gamma ** (epoch // self.step_size)
        return self.initial_lr * decay_factor


class CosineAnnealingScheduler(LearningRateScheduler):
    """Cosine annealing learning rate scheduler.
    
    Smoothly decreases learning rate following cosine curve:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * epoch / T_max))
    
    Args:
        initial_lr: Maximum learning rate
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate (default: 0)
    """
    
    def __init__(
        self,
        initial_lr: float,
        T_max: int,
        eta_min: float = 0.0,
    ):
        self.lr_max = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for epoch."""
        if epoch >= self.T_max:
            return self.eta_min
        
        cosine_term = 0.5 * (1 + np.cos(np.pi * epoch / self.T_max))
        lr = self.eta_min + (self.lr_max - self.eta_min) * cosine_term
        
        return float(lr)


class WarmupScheduler(LearningRateScheduler):
    """Warmup scheduler — gradually increases learning rate.
    
    Linear warmup followed by constant or decay:
        lr = lr_0 * min(1, epoch / warmup_epochs)
    
    Args:
        initial_lr: Target learning rate after warmup
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Starting learning rate (default: 0)
    """
    
    def __init__(
        self,
        initial_lr: float,
        warmup_epochs: int,
        warmup_start_lr: float = 0.0,
    ):
        self.target_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for epoch."""
        if epoch >= self.warmup_epochs:
            return self.target_lr
        
        # Linear warmup
        progress = epoch / self.warmup_epochs
        lr = self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * progress
        
        return float(lr)
