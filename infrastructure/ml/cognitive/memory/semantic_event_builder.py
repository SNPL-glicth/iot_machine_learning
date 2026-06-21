"""
SemanticEventBuilder for constructing semantic operational events.

Builds MemoryEvent objects from operational data with semantic text generation.
"""

from typing import Optional, Dict, Any

from domain.entities.memory import MemoryEvent


class SemanticEventBuilder:
    """Builder for semantic operational events."""
    
    def build(
        self,
        sensor_id: int,
        sensor_type: str,
        ml_features: Dict[str, Any],
        regime: str,
        anomaly_score: float,
        previous_regime: Optional[str] = None,
        transition_duration: Optional[float] = None,
    ) -> MemoryEvent:
        """
        Build semantic event from operational data.
        
        Args:
            sensor_id: Sensor identifier
            sensor_type: Sensor type (e.g., "TEMPERATURE", "PRESSURE")
            ml_features: ML features dictionary
            regime: Current operational regime
            anomaly_score: Anomaly score
            previous_regime: Previous regime (if available)
            transition_duration: Duration of transition (if available)
        
        Returns:
            MemoryEvent with semantic text and metadata
        """
        # Determine event type
        event_type = self._classify_event_type(regime, anomaly_score, previous_regime)
        
        # Generate semantic text
        semantic_text = self._generate_semantic_text(
            sensor_id, sensor_type, ml_features, regime, anomaly_score
        )
        
        # Build metadata
        metadata = self._build_metadata(ml_features, previous_regime, transition_duration)
        
        # Extract dynamic features
        dynamic_features = ml_features.get("dynamic_features", {})
        
        return MemoryEvent(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            timestamp=ml_features.get("timestamp", 0.0),
            event_type=event_type,
            semantic_text=semantic_text,
            regime=regime,
            anomaly_score=anomaly_score,
            dynamic_features=dynamic_features,
            metadata=metadata,
        )
    
    def _classify_event_type(
        self,
        regime: str,
        anomaly_score: float,
        previous_regime: Optional[str],
    ) -> str:
        """Classify event type."""
        if anomaly_score > 0.8:
            return "ANOMALY_CONFIRMED"
        elif anomaly_score > 0.6:
            return "ANOMALY_SUSPECTED"
        elif previous_regime and regime != previous_regime:
            return "REGIME_TRANSITION"
        elif regime in ["STARTUP", "SHUTDOWN"]:
            return "OPERATIONAL_TRANSITION"
        else:
            return "OPERATIONAL_STATE"
    
    def _generate_semantic_text(
        self,
        sensor_id: int,
        sensor_type: str,
        ml_features: Dict[str, Any],
        regime: str,
        anomaly_score: float,
    ) -> str:
        """Generate semantic text for embedding."""
        current_value = ml_features.get("current_value", 0.0)
        baseline = ml_features.get("baseline", 0.0)
        z_score = ml_features.get("z_score", 0.0)
        
        text = f"Sensor {sensor_id} ({sensor_type}) en régimen {regime}. "
        text += f"Valor actual: {current_value:.2f}, Baseline: {baseline:.2f}, Z-score: {z_score:.2f}. "
        
        if anomaly_score > 0.6:
            text += f"Anomalía score: {anomaly_score:.2f}. "
        
        # Add dynamic features if available
        dynamic_features = ml_features.get("dynamic_features", {})
        if dynamic_features:
            derivative = dynamic_features.get("derivative")
            if derivative is not None:
                text += f"Derivada: {derivative:.2f}. "
            
            rolling_std = dynamic_features.get("rolling_std_1h")
            if rolling_std is not None:
                text += f"Volatilidad (1h): {rolling_std:.2f}. "
        
        return text
    
    def _build_metadata(
        self,
        ml_features: Dict[str, Any],
        previous_regime: Optional[str],
        transition_duration: Optional[float],
    ) -> Dict[str, Any]:
        """Build structured metadata."""
        return {
            "value": ml_features.get("current_value", 0.0),
            "baseline": ml_features.get("baseline", 0.0),
            "z_score": ml_features.get("z_score", 0.0),
            "trend": ml_features.get("trend", "unknown"),
            "stability": ml_features.get("stability", 0.0),
            "previous_regime": previous_regime,
            "transition_duration": transition_duration,
            "model_version": ml_features.get("model_version", "2.0.0"),
        }
