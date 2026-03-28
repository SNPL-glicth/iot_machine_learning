"""WeightResolutionService — consolidated weight resolution for orchestrator.

Replaces the dual WeightAdjustmentService + WeightMediator pattern with a single
component that handles all weight resolution logic.

Phase 3 refactor: Simplifies MetaCognitiveOrchestrator by extracting
weight-related concerns into this cohesive service.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WeightResolutionService:
    """Consolidated service for resolving engine fusion weights.
    
    Combines responsibilities of WeightAdjustmentService and WeightMediator
    into a single, testable component.
    
    Responsibilities:
    - Priority-based weight resolution (adaptive → plasticity → base)
    - Weight normalization and validation
    - Fallback handling for missing data
    
    Design:
    - Stateless: no internal mutable state
    - Fail-safe: returns base weights on any error
    - Testable: pure logic, dependencies injected
    """

    def __init__(
        self,
        base_weights: Dict[str, float],
        plasticity_tracker=None,
        storage_adapter=None,
        epsilon: float = 0.01,
    ) -> None:
        """Initialize weight resolution service.
        
        Args:
            base_weights: Default weights (fallback when no learning data)
            plasticity_tracker: Optional plasticity for learned weights
            storage_adapter: Optional storage for rolling MAE data
            epsilon: Small value to prevent division by zero
        """
        self._base_weights = base_weights
        self._plasticity = plasticity_tracker
        self._storage = storage_adapter
        self._epsilon = epsilon

    def resolve(
        self,
        regime: str,
        engine_names: List[str],
        series_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Resolve weights for engines using priority cascade.
        
        Priority (highest first):
        1. Adaptive weights from rolling MAE (if storage available and data exists)
        2. Plasticity weights (if tracker has history for regime)
        3. Base weights (guaranteed fallback)
        
        Args:
            regime: Current signal regime label
            engine_names: List of engine names to get weights for
            series_id: Optional series identifier for storage queries
            
        Returns:
            Dict mapping engine_name → normalized weight (sums to 1.0)
        """
        if not engine_names:
            return {}

        # Priority 1: Try adaptive weights from storage (rolling MAE)
        if self._storage is not None and series_id is not None:
            try:
                adaptive = self._compute_adaptive_weights(series_id, engine_names)
                if adaptive and len(adaptive) == len(engine_names):
                    return self._normalize(adaptive, engine_names)
            except Exception as e:
                logger.debug(f"adaptive_weights_failed: {e}")

        # Priority 2: Try plasticity weights if tracker has history
        if self._plasticity is not None:
            try:
                if hasattr(self._plasticity, 'has_history') and \
                   hasattr(self._plasticity, 'get_weights'):
                    if self._plasticity.has_history(regime):
                        plasticity_weights = self._plasticity.get_weights(
                            regime, engine_names
                        )
                        if plasticity_weights:
                            return self._normalize(plasticity_weights, engine_names)
            except Exception as e:
                logger.debug(f"plasticity_weights_failed: {e}")

        # Priority 3: Base weights (guaranteed fallback)
        return self._get_base_weights(engine_names)

    def _compute_adaptive_weights(
        self,
        series_id: str,
        engine_names: List[str],
    ) -> Optional[Dict[str, float]]:
        """Compute weights from rolling MAE in storage.
        
        Args:
            series_id: Series identifier
            engine_names: Engines to compute weights for
            
        Returns:
            Dict of weights or None if data unavailable
        """
        # Check if storage has required method
        if not hasattr(self._storage, 'get_rolling_mae'):
            return None

        mae_values: Dict[str, float] = {}
        for name in engine_names:
            try:
                mae = self._storage.get_rolling_mae(series_id, name)
                if mae is not None and mae > 0:
                    mae_values[name] = mae
            except Exception:
                continue

        if len(mae_values) < len(engine_names):
            return None

        # Convert MAE to weights (inverse, normalized)
        # Lower MAE = higher weight
        inv_mae = {name: 1.0 / (mae + self._epsilon) for name, mae in mae_values.items()}
        total = sum(inv_mae.values())
        
        if total < self._epsilon:
            return None
            
        return {name: w / total for name, w in inv_mae.items()}

    def _get_base_weights(self, engine_names: List[str]) -> Dict[str, float]:
        """Get base weights for engines.
        
        Falls back to uniform if base weights incomplete.
        """
        weights: Dict[str, float] = {}
        for name in engine_names:
            weights[name] = self._base_weights.get(name, 1.0 / len(engine_names))
        return self._normalize(weights, engine_names)

    def _normalize(
        self,
        weights: Dict[str, float],
        engine_names: List[str],
    ) -> Dict[str, float]:
        """Normalize weights to sum to 1.0.
        
        Handles edge cases:
        - Missing engines get uniform weight
        - Zero/negative weights replaced with epsilon
        - Empty weights fall back to uniform
        """
        # Ensure all engines have a weight
        complete: Dict[str, float] = {}
        for name in engine_names:
            w = weights.get(name, 0.0)
            if w <= 0:
                w = self._epsilon
            complete[name] = w

        total = sum(complete.values())
        if total < self._epsilon:
            # Fallback to uniform
            uniform = 1.0 / len(engine_names)
            return {name: uniform for name in engine_names}

        return {name: w / total for name, w in complete.items()}
