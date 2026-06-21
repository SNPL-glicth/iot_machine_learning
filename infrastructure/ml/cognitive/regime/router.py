"""
ContextualAnomalyRouter for routing anomalies with regime context.

Applies regime-specific thresholds and multipliers to anomaly scores.
"""

from typing import Dict, Optional
from dataclasses import dataclass

from .models.anomaly_thresholds import AnomalyThresholds
from .models.regime_config import RegimeConfig


@dataclass
class ContextualAnomalyResult:
    """Result of contextual anomaly routing."""
    sensor_id: int
    regime: str
    base_score: float
    contextual_score: float
    is_anomalous: bool
    threshold_used: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sensor_id": self.sensor_id,
            "regime": self.regime,
            "base_score": self.base_score,
            "contextual_score": self.contextual_score,
            "is_anomalous": self.is_anomalous,
            "threshold_used": self.threshold_used,
        }


class ContextualAnomalyRouter:
    """Router for anomalies with regime context."""
    
    def __init__(
        self,
        regime_thresholds: Optional[Dict[str, AnomalyThresholds]] = None,
    ):
        """
        Initialize contextual anomaly router.
        
        Args:
            regime_thresholds: Dictionary mapping regime name to thresholds
        """
        self._thresholds = regime_thresholds or {}
        self._default_thresholds = AnomalyThresholds.default()
    
    def route_anomaly(
        self,
        sensor_id: int,
        current_value: float,
        regime: str,
        base_anomaly_score: float,
    ) -> ContextualAnomalyResult:
        """
        Route anomaly with regime context.
        
        Args:
            sensor_id: Sensor identifier
            current_value: Current sensor value
            regime: Current operational regime
            base_anomaly_score: Base anomaly score from detectors
        
        Returns:
            Contextual anomaly result
        """
        # Get thresholds for regime (fallback to default)
        thresholds = self._thresholds.get(regime, self._default_thresholds)
        
        # Adjust score according to regime
        contextual_score = self._adjust_score_by_regime(
            base_score=base_anomaly_score,
            regime=regime,
            thresholds=thresholds,
        )
        
        # Determine if anomalous in context
        is_anomalous = contextual_score > thresholds.anomaly_threshold
        
        return ContextualAnomalyResult(
            sensor_id=sensor_id,
            regime=regime,
            base_score=base_anomaly_score,
            contextual_score=contextual_score,
            is_anomalous=is_anomalous,
            threshold_used=thresholds.anomaly_threshold,
        )
    
    def _adjust_score_by_regime(
        self,
        base_score: float,
        regime: str,
        thresholds: AnomalyThresholds,
    ) -> float:
        """
        Adjust anomaly score according to regime.
        
        Args:
            base_score: Base anomaly score
            regime: Current regime
            thresholds: Anomaly thresholds
        
        Returns:
            Contextually adjusted score
        """
        if regime == "VOLATILE_PEAK":
            # In peak load, reduce sensitivity
            return base_score * thresholds.peak_load_multiplier
        elif regime == "STARTUP" or regime == "SHUTDOWN":
            # In transitions, maintain sensitivity
            return base_score * thresholds.transition_multiplier
        elif regime == "STABLE_NORMAL":
            # In steady-state, normal sensitivity
            return base_score * thresholds.normal_multiplier
        elif regime == "ANOMALOUS_REGIME":
            # In anomalous regime, maximum sensitivity
            return base_score * 1.0
        else:
            return base_score
