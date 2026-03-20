"""Metaplasticity — plasticity of plasticity (BCM theory).

Bienenstock-Cooper-Munro (BCM) theory:
The synaptic modification threshold θ_M slides based on postsynaptic activity.
This prevents runaway potentiation and enables stable learning.

θ_M(t) = θ_M(t-1) + (activity² - θ_M) / τ_BCM
"""

from __future__ import annotations

import numpy as np
from typing import Dict


class MetaplasticityController:
    """BCM sliding threshold for adaptive learning.
    
    Controls when synaptic weights should be strengthened vs weakened
    based on history of postsynaptic activity.
    
    Args:
        tau_BCM: Time constant for threshold sliding (ms)
        initial_threshold: Initial modification threshold
        min_threshold: Minimum threshold value
        max_threshold: Maximum threshold value
    """
    
    def __init__(
        self,
        tau_BCM: float = 1000.0,
        initial_threshold: float = 0.5,
        min_threshold: float = 0.1,
        max_threshold: float = 2.0,
    ) -> None:
        self.tau_BCM = tau_BCM
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Per-domain thresholds
        self.thresholds: Dict[str, float] = {}
        self.initial_threshold = initial_threshold
    
    def update_threshold(
        self,
        domain: str,
        current_activity: float,
        dt: float = 1.0,
    ) -> float:
        """Update modification threshold for domain.
        
        BCM rule: θ_M moves toward activity² to track history.
        
        Args:
            domain: Domain identifier
            current_activity: Current postsynaptic activity [0, 1]
            dt: Time step (ms)
            
        Returns:
            Updated threshold value
        """
        # Get or initialize threshold for this domain
        if domain not in self.thresholds:
            self.thresholds[domain] = self.initial_threshold
        
        theta_current = self.thresholds[domain]
        
        # BCM sliding threshold update
        # θ_M moves toward activity² with time constant τ_BCM
        activity_squared = current_activity ** 2
        d_theta = (activity_squared - theta_current) / self.tau_BCM
        
        theta_new = theta_current + d_theta * dt
        
        # Apply bounds
        theta_new = np.clip(theta_new, self.min_threshold, self.max_threshold)
        
        self.thresholds[domain] = theta_new
        
        return theta_new
    
    def get_threshold(self, domain: str) -> float:
        """Get current threshold for domain.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Current modification threshold
        """
        return self.thresholds.get(domain, self.initial_threshold)
    
    def should_potentiate(
        self,
        domain: str,
        activity: float,
    ) -> bool:
        """Determine if weights should be strengthened.
        
        BCM rule: potentiate if activity > threshold
        
        Args:
            domain: Domain identifier
            activity: Current activity level
            
        Returns:
            True if should potentiate (strengthen weights)
        """
        threshold = self.get_threshold(domain)
        return activity > threshold
    
    def compute_learning_factor(
        self,
        domain: str,
        activity: float,
    ) -> float:
        """Compute learning rate modulation factor.
        
        Returns positive for potentiation, negative for depression.
        
        Args:
            domain: Domain identifier
            activity: Current activity level
            
        Returns:
            Learning factor [-1, 1]
        """
        threshold = self.get_threshold(domain)
        
        # Signed distance from threshold
        # Positive → potentiation, Negative → depression
        factor = (activity - threshold) / (threshold + 1e-9)
        
        return np.clip(factor, -1.0, 1.0)
    
    def reset_domain(self, domain: str) -> None:
        """Reset threshold for domain to initial value.
        
        Args:
            domain: Domain identifier
        """
        self.thresholds[domain] = self.initial_threshold
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all domain thresholds.
        
        Returns:
            Dict of {domain: threshold}
        """
        return self.thresholds.copy()
