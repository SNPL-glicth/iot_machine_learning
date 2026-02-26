from __future__ import annotations

from typing import Dict, List, Tuple


class GradientPropagationService:
    """Propagates gradients through correlation network."""
    
    @staticmethod
    def compute_propagated_gradient(
        series_id: str,
        local_gradient: float,
        neighbors: List[Tuple[str, float]],
        neighbor_gradients: Dict[str, float],
        propagation_strength: float = 0.3,
    ) -> float:
        """Compute propagated gradient from correlated neighbors.
        
        Formula: gradient_total = gradient_local + α·Σ(corr_i × gradient_i) / Σ|corr_i|
        
        Args:
            series_id: Series identifier
            local_gradient: Local gradient (OLS over window)
            neighbors: List of (neighbor_id, correlation) tuples
            neighbor_gradients: Dict of neighbor gradients
            propagation_strength: α ∈ [0, 1] (default 0.3)
        
        Returns:
            Propagated gradient
        """
        if not neighbors:
            return local_gradient
        
        propagated_component = 0.0
        total_abs_correlation = 0.0
        
        for neighbor_id, correlation in neighbors:
            if abs(correlation) > 0.5 and neighbor_id in neighbor_gradients:
                propagated_component += correlation * neighbor_gradients[neighbor_id]
                total_abs_correlation += abs(correlation)
        
        if total_abs_correlation < 1e-9:
            return local_gradient
        
        propagated_component /= total_abs_correlation
        
        return local_gradient + propagation_strength * propagated_component
