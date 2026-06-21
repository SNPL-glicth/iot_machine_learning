"""
OperationalSummaryBuilder for building operational summaries.

Builds structured operational summaries from contextual explanations.
"""

from typing import Dict, Any

from domain.entities.explainability import ContextualExplanation


class OperationalSummaryBuilder:
    """Builder for operational summaries."""
    
    def build(self, explanation: ContextualExplanation) -> Dict[str, Any]:
        """
        Build operational summary from contextual explanation.
        
        Args:
            explanation: ContextualExplanation
        
        Returns:
            Dictionary with operational summary
        """
        summary = {
            "sensor_id": explanation.sensor_id,
            "sensor_type": explanation.sensor_type,
            "timestamp": explanation.timestamp,
            "current_regime": explanation.current_regime,
            "anomaly_score": explanation.anomaly_score,
            "operational_confidence": explanation.operational_confidence,
            "primary_drivers": explanation.primary_drivers,
            "similar_event_count": explanation.similar_event_count,
            "suggested_actions": explanation.suggested_actions,
            "summary_text": self._generate_summary_text(explanation),
        }
        
        return summary
    
    def _generate_summary_text(self, explanation: ContextualExplanation) -> str:
        """Generate summary text from explanation."""
        text = f"Sensor {explanation.sensor_id} ({explanation.sensor_type}) "
        text += f"operando en régimen {explanation.current_regime}. "
        
        if explanation.anomaly_score > 0.6:
            text += f"Anomalía detectada con score {explanation.anomaly_score:.2f}. "
            text += f"Drivers principales: {', '.join(explanation.primary_drivers)}. "
        else:
            text += f"Operación normal con score de anomalía {explanation.anomaly_score:.2f}. "
        
        if explanation.similar_event_count > 0:
            text += f"{explanation.historical_context} "
        
        if explanation.suggested_actions:
            text += f"Acciones sugeridas: {', '.join(explanation.suggested_actions)}."
        
        return text
