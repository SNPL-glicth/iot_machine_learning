"""
Propagation confidence utilities for event propagation tracker.

Provides confidence calculation for propagation events.
"""


class PropagationConfidenceCalculator:
    """Calculator for propagation confidence."""
    
    @staticmethod
    def calculate(
        target_count: int,
        duration: float,
        max_window_seconds: float,
    ) -> float:
        """Calculate propagation confidence."""
        target_factor = min(1.0, target_count / 5.0)
        duration_factor = max(0.0, 1.0 - (duration / max_window_seconds))
        
        return (target_factor * 0.6 + duration_factor * 0.4)
