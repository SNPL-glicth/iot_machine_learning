"""Hampel Fallback Strategy (COG-CRIT-2).

Strategy for selecting engine when Hampel filter rejects all predictions.

Applies OCP: Fallback logic is injectable strategy, not hardcoded.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from ...analysis.types import EnginePerception, InhibitionState

logger = logging.getLogger(__name__)


class HampelFallbackStrategy(ABC):
    """Abstract strategy for Hampel fallback (COG-CRIT-2).
    
    Applies OCP: Different fallback strategies can be implemented
    without modifying FusePhase.
    """
    
    @abstractmethod
    def select_fallback(
        self,
        perceptions: List[EnginePerception],
        inhibition_states: List[InhibitionState],
        median: float,
    ) -> Tuple[List[EnginePerception], List[InhibitionState], str]:
        """Select fallback when Hampel rejects all engines.
        
        Args:
            perceptions: All perceptions (rejected by Hampel).
            inhibition_states: Corresponding inhibition states.
            median: Median of rejected predictions.
        
        Returns:
            Tuple of (selected_perceptions, selected_states, reason).
        
        Applies SRP: Only selection logic, no filtering.
        """
        pass


class MedianClosestFallbackStrategy(HampelFallbackStrategy):
    """Select engine closest to median when Hampel rejects all (COG-CRIT-2).
    
    Rationale: If all predictions are outliers, the one closest to median
    is the most conservative choice.
    
    Applies OCP: Concrete strategy implementation.
    """
    
    def select_fallback(
        self,
        perceptions: List[EnginePerception],
        inhibition_states: List[InhibitionState],
        median: float,
    ) -> Tuple[List[EnginePerception], List[InhibitionState], str]:
        """Select engine with prediction closest to median.
        
        Args:
            perceptions: All perceptions (rejected by Hampel).
            inhibition_states: Corresponding inhibition states.
            median: Median of rejected predictions.
        
        Returns:
            Tuple of ([closest_perception], [closest_state], reason).
        
        Applies COG-CRIT-2: Median-based fallback instead of bypass.
        """
        if not perceptions:
            return [], [], "no_perceptions_available"
        
        # Find perception closest to median
        closest_perception = min(
            perceptions,
            key=lambda p: abs(p.predicted_value - median)
        )
        
        # Find corresponding inhibition state
        closest_state = None
        for state in inhibition_states:
            if state.engine_name == closest_perception.engine_name:
                closest_state = state
                break
        
        # If no matching state, create default one
        if closest_state is None:
            closest_state = InhibitionState(
                engine_name=closest_perception.engine_name,
                base_weight=1.0 / len(perceptions),  # Equal weight fallback
                inhibited_weight=1.0 / len(perceptions),
                inhibition_reason="none",
                suppression_factor=0.0,
            )
        
        reason = (
            f"hampel_all_rejected_using_median:"
            f"selected_{closest_perception.engine_name}_"
            f"closest_to_median_{median:.2f}"
        )
        
        logger.info(
            "hampel_fallback_median_closest",
            extra={
                "selected_engine": closest_perception.engine_name,
                "predicted_value": round(closest_perception.predicted_value, 4),
                "median": round(median, 4),
                "distance_from_median": round(
                    abs(closest_perception.predicted_value - median), 4
                ),
                "n_rejected": len(perceptions),
            },
        )
        
        return [closest_perception], [closest_state], reason


class BypassAllFallbackStrategy(HampelFallbackStrategy):
    """Legacy strategy: bypass Hampel when all rejected (COG-CRIT-2).
    
    This is the old behavior - kept for backward compatibility.
    Not recommended for production use.
    """
    
    def select_fallback(
        self,
        perceptions: List[EnginePerception],
        inhibition_states: List[InhibitionState],
        median: float,
    ) -> Tuple[List[EnginePerception], List[InhibitionState], str]:
        """Bypass Hampel and return all perceptions.
        
        Args:
            perceptions: All perceptions (rejected by Hampel).
            inhibition_states: Corresponding inhibition states.
            median: Median of rejected predictions (unused).
        
        Returns:
            Tuple of (all_perceptions, all_states, reason).
        
        Legacy behavior: Not recommended.
        """
        logger.warning(
            "hampel_fallback_bypass_all",
            extra={
                "n_perceptions": len(perceptions),
                "median": round(median, 4),
                "warning": "Using legacy bypass strategy - consider MedianClosestFallbackStrategy",
            },
        )
        
        return (
            list(perceptions),
            list(inhibition_states),
            "hampel_all_rejected_bypassed",
        )
