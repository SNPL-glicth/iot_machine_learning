"""Homeostatic plasticity — maintains target activity levels.

Prevents runaway excitation/inhibition by scaling synaptic weights
to maintain target firing rate. This is a slow adaptation mechanism
that works over longer timescales than STDP.

Synaptic scaling: w → w * (target_activity / current_activity)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class HomeostaticRegulator:
    """Homeostatic synaptic scaling for stable network activity.
    
    Maintains target activity level by multiplicatively scaling
    all incoming weights to a neuron.
    
    Args:
        target_activity: Target firing rate (fraction active)
        tau_homeostatic: Time constant for homeostatic changes (ms)
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor
    """
    
    def __init__(
        self,
        target_activity: float = 0.1,
        tau_homeostatic: float = 100000.0,  # Very slow (100 seconds)
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ) -> None:
        self.target_activity = target_activity
        self.tau_homeostatic = tau_homeostatic
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def regulate(
        self,
        weights: np.ndarray,
        current_activity: float,
        dt: float = 1.0,
    ) -> np.ndarray:
        """Apply homeostatic scaling to weights.
        
        Scales all weights to move current activity toward target.
        
        Args:
            weights: Synaptic weight matrix
            current_activity: Current firing rate (fraction active)
            dt: Time step (ms)
            
        Returns:
            Scaled weights
        """
        if current_activity < 1e-9:
            # No activity - don't scale
            return weights
        
        # Compute scaling factor
        # If activity too high → scale down
        # If activity too low → scale up
        target_scale = self.target_activity / current_activity
        
        # Apply time constant (slow adaptation)
        alpha = dt / self.tau_homeostatic
        scale_factor = 1.0 + alpha * (target_scale - 1.0)
        
        # Apply bounds
        scale_factor = np.clip(scale_factor, self.min_scale, self.max_scale)
        
        # Scale weights
        scaled_weights = weights * scale_factor
        
        return scaled_weights
    
    def compute_scaling_factor(
        self,
        current_activity: float,
        dt: float = 1.0,
    ) -> float:
        """Compute scaling factor without applying to weights.
        
        Args:
            current_activity: Current firing rate
            dt: Time step (ms)
            
        Returns:
            Scaling factor to apply
        """
        if current_activity < 1e-9:
            return 1.0
        
        target_scale = self.target_activity / current_activity
        alpha = dt / self.tau_homeostatic
        scale_factor = 1.0 + alpha * (target_scale - 1.0)
        
        return np.clip(scale_factor, self.min_scale, self.max_scale)
    
    def is_activity_in_range(
        self,
        current_activity: float,
        tolerance: float = 0.2,
    ) -> bool:
        """Check if activity is within acceptable range.
        
        Args:
            current_activity: Current firing rate
            tolerance: Acceptable deviation from target (fraction)
            
        Returns:
            True if activity is within target ± tolerance
        """
        lower_bound = self.target_activity * (1.0 - tolerance)
        upper_bound = self.target_activity * (1.0 + tolerance)
        
        return lower_bound <= current_activity <= upper_bound
    
    def adjust_target(
        self,
        new_target: float,
    ) -> None:
        """Adjust target activity level.
        
        Args:
            new_target: New target activity [0, 1]
        """
        self.target_activity = np.clip(new_target, 0.01, 0.5)
