"""Weight Adjustment Service.

Handles adaptive weight calculation based on rolling MAE and plasticity history.
Extracted from MetaCognitiveOrchestrator to reduce complexity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WeightAdjustmentService:
    """Manages adaptive weight calculation for engine fusion.
    
    Responsibilities:
    - Compute adaptive weights from rolling MAE
    - Fallback to plasticity weights
    - Fallback to uniform weights
    - Weight normalization
    """
    
    def __init__(
        self,
        base_weights: Dict[str, float],
        storage_adapter=None,
        plasticity_tracker=None,
        epsilon: float = 0.01,
    ):
        """Initialize weight adjustment service.
        
        Args:
            base_weights: Base weights for engines
            storage_adapter: Optional storage for rolling MAE
            plasticity_tracker: Optional PlasticityTracker
            epsilon: Small value to prevent division by zero
        """
        self._base_weights = base_weights
        self._storage = storage_adapter
        self._plasticity = plasticity_tracker
        self._epsilon = epsilon
    
    def resolve_weights(
        self,
        regime: str,
        engine_names: List[str],
        series_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Resolve weights for engines using adaptive strategy.
        
        Priority:
        1. Adaptive weights from rolling MAE (if storage available)
        2. Plasticity weights (if history available)
        3. Base/uniform weights
        
        Args:
            regime: Signal regime string
            engine_names: List of engine names
            series_id: Optional series identifier for adaptive weights
        
        Returns:
            Dict mapping engine_name to weight (normalized to sum to 1)
        """
        # Try adaptive weights based on rolling MAE
        if self._storage and series_id:
            adaptive_weights = self._compute_adaptive_weights(series_id, engine_names)
            if adaptive_weights:
                return adaptive_weights
        
        # Fallback: plasticity or uniform weights
        if self._plasticity and self._plasticity.has_history(regime):
            return self._plasticity.get_weights(regime, engine_names)
        
        # Final fallback: base or uniform weights
        return {n: self._base_weights.get(n, 1.0 / len(engine_names)) for n in engine_names}
    
    def _compute_adaptive_weights(
        self,
        series_id: str,
        engine_names: List[str],
    ) -> Optional[Dict[str, float]]:
        """Compute adaptive weights based on rolling MAE.
        
        Formula: weight = 1 / (rolling_mae + epsilon)
        Normalized so weights sum to 1.
        
        Args:
            series_id: Series identifier
            engine_names: List of engine names
        
        Returns:
            Normalized weights dict, or None if insufficient data
        """
        if not self._storage:
            return None
        
        raw_weights = {}
        for engine_name in engine_names:
            perf = self._storage.get_rolling_performance(series_id, engine_name)
            if not perf:
                # Not enough historical data for this engine
                return None
            
            mae = perf["mae"]
            # Inverse MAE: lower error = higher weight
            raw_weights[engine_name] = 1.0 / (mae + self._epsilon)
        
        # Normalize to sum to 1
        total = sum(raw_weights.values())
        if total < 1e-9:
            return None
        
        normalized = {k: v / total for k, v in raw_weights.items()}
        
        logger.debug(
            "adaptive_weights_computed",
            extra={"series_id": series_id, "weights": normalized},
        )
        
        return normalized
