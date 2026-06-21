"""
Cross-feature generator for dynamic feature computation.

Generates cross-features between correlated sensors (e.g., pressure × temperature).
"""

from typing import Dict, Optional


class CrossFeatureGenerator:
    """Generates cross-features between correlated sensors."""
    
    def __init__(
        self,
        correlation_threshold: float = 0.7,
        max_cross_features: int = 5,
    ):
        """
        Initialize cross-feature generator.
        
        Args:
            correlation_threshold: Minimum correlation to consider cross-features
            max_cross_features: Maximum number of cross-features per sensor
        """
        self._correlation_threshold = correlation_threshold
        self._max_cross_features = max_cross_features
    
    def compute_cross_features(
        self,
        sensor_id: int,
        current_value: float,
        value_store: Dict[int, float],
        correlations: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute cross-features with correlated sensors.
        
        Args:
            sensor_id: Current sensor ID
            current_value: Current sensor value
            value_store: Dictionary mapping sensor_id -> latest value
            correlations: Dictionary mapping other_sensor_id -> correlation coefficient.
                          If None, assumes all sensors in value_store are correlated.
        
        Returns:
            Dictionary mapping cross_feature_name -> cross_feature_value
        """
        cross_features = {}
        
        # If no correlations provided, use all sensors in value_store
        if correlations is None:
            correlations = {sid: 1.0 for sid in value_store.keys() if sid != sensor_id}
        
        # Sort by correlation (highest first) and take top N
        sorted_correlations = sorted(
            correlations.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:self._max_cross_features]
        
        for other_sensor_id, correlation in sorted_correlations:
            # Only use strong correlations
            if correlation >= self._correlation_threshold:
                other_value = value_store.get(other_sensor_id)
                
                if other_value is not None:
                    # Cross-feature: simple product
                    cross_name = f"cross_{sensor_id}_{other_sensor_id}"
                    cross_features[cross_name] = current_value * other_value
        
        return cross_features
    
    def compute_single_cross_feature(
        self,
        value1: float,
        value2: float,
    ) -> float:
        """
        Compute a single cross-feature (product).
        
        Args:
            value1: First value
            value2: Second value
        
        Returns:
            Cross-feature value (product)
        """
        return value1 * value2
