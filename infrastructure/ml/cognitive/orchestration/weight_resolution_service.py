"""WeightResolutionService — consolidated weight resolution for orchestrator.

Replaces the dual WeightAdjustmentService + WeightMediator pattern with a single
component that handles all weight resolution logic.

Phase 3 refactor: Simplifies MetaCognitiveOrchestrator by extracting
weight-related concerns into this cohesive service.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from core.parameters.numerical_constants import EPSILON

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
    
    CRIT-3 FIX: EPSILON value restored to 0.01 for numerical stability.
    Previous change to EPSILON.DIVISION (1e-12) introduced risk of weight divergence
    when MAE values are very small. The 0.01 value provides a safer floor while still
    being small enough to not significantly impact weight calculations.
    """

    # CRIT-3 FIX: Safe default epsilon for numerical stability
    # 0.01 provides a stable floor for division operations while being small enough
    # to not significantly impact weight calculations in normal scenarios.
    _DEFAULT_EPSILON: float = 0.01

    def __init__(
        self,
        base_weights: Dict[str, float],
        plasticity_tracker=None,
        storage_adapter=None,
        epsilon: Optional[float] = None,
    ) -> None:
        """Initialize weight resolution service.
        
        Args:
            base_weights: Default weights (fallback when no learning data)
            plasticity_tracker: Optional plasticity for learned weights
            storage_adapter: Optional storage for rolling MAE data
            epsilon: Small value to prevent division by zero. If None, uses 0.01.
                CRIT-3 FIX: Default restored to 0.01 from 1e-12 for numerical stability.
                Values < 1e-6 trigger a warning due to potential numerical instability.
        """
        # CRIT-3 FIX: Use safe default epsilon if not provided
        if epsilon is None:
            epsilon = self._DEFAULT_EPSILON
        
        # CRIT-3 FIX: Warn if epsilon is too small (potential numerical instability)
        if epsilon < 1e-6:
            logger.warning(
                "weight_resolution_epsilon_too_small",
                extra={
                    "epsilon": epsilon,
                    "recommended_min": 1e-6,
                    "risk": "numerical_instability_division_near_zero",
                },
            )
        
        # CRIT-3 FIX: Validate epsilon is positive
        if epsilon <= 0:
            raise ValueError(
                f"epsilon must be positive, received {epsilon}"
            )
        self._base_weights = base_weights
        self._plasticity = plasticity_tracker
        self._storage = storage_adapter
        self._epsilon = epsilon
        
        # FASE-22: Validate base_weights sum ≈ 1.0 (warn if misconfigured)
        if base_weights:
            weights_sum = sum(base_weights.values())
            if not math.isclose(weights_sum, 1.0, rel_tol=EPSILON.COMPARISON):
                logger.warning(
                    "base_weights_sum_drift",
                    extra={
                        "sum": weights_sum,
                        "expected": 1.0,
                        "engines": list(base_weights.keys()),
                        "note": "base_weights will be normalized automatically"
                    }
                )

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
        
        normalized = {name: w / total for name, w in inv_mae.items()}
        
        # Validate normalization: sum should be ≈ 1.0
        normalized_sum = sum(normalized.values())
        if not math.isclose(normalized_sum, 1.0, rel_tol=EPSILON.COMPARISON):
            logger.warning(
                "adaptive_mae_weights_normalization_drift",
                extra={"sum": normalized_sum, "expected": 1.0}
            )
        
        return normalized

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

        normalized = {name: w / total for name, w in complete.items()}
        
        # Validate normalization: sum should be ≈ 1.0
        normalized_sum = sum(normalized.values())
        if not math.isclose(normalized_sum, 1.0, rel_tol=EPSILON.COMPARISON):
            logger.warning(
                "weight_normalization_drift",
                extra={"sum": normalized_sum, "expected": 1.0, "engines": engine_names}
            )
        
        return normalized
