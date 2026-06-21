"""
Heuristic classifier for regime detection (fallback).
"""

from typing import TYPE_CHECKING

from .models.regime_prediction import RegimePrediction
from .models.regime_config import RegimeConfig

if TYPE_CHECKING:
    from infrastructure.ml.cognitive.dynamic.models.dynamic_features import DynamicFeatures


class HeuristicClassifier:
    """Heuristic classifier based on rules (fallback)."""
    
    def classify(
        self,
        dynamic_features: 'DynamicFeatures',
        config: RegimeConfig,
        current_value: float = None,
    ) -> RegimePrediction:
        """
        Classify using heuristic rules.
        
        Args:
            dynamic_features: Dynamic features
            config: Regime configuration
            current_value: Current sensor value
        
        Returns:
            Regime prediction
        """
        derivative = dynamic_features.derivative or 0.0
        rolling_std = dynamic_features.rolling_std_1h or 0.0
        
        if abs(derivative) > config.derivative_threshold_high:
            if derivative > 0:
                regime = "STARTUP"
            else:
                regime = "SHUTDOWN"
        elif rolling_std > config.volatility_threshold_high:
            regime = "VOLATILE_PEAK"
        elif rolling_std < config.volatility_threshold_low:
            if current_value and current_value < config.low_threshold:
                regime = "STABLE_LOW"
            elif current_value and current_value > config.high_threshold:
                regime = "STABLE_HIGH"
            else:
                regime = "STABLE_NORMAL"
        else:
            regime = "STABLE_NORMAL"
        
        return RegimePrediction(regime=regime, confidence=0.7, cluster_idx=-1)
