"""
RecommendationGenerator for generating simple heuristic recommendations.

Generates operational recommendations based on context and patterns.
"""

from typing import List, Dict, Any


class RecommendationGenerator:
    """Generator for simple heuristic recommendations."""
    
    def generate(
        self,
        regime: str,
        anomaly_score: float,
        dynamic_features: Dict[str, Any],
        historical_patterns: List[str],
    ) -> List[str]:
        """
        Generate heuristic recommendations.
        
        Args:
            regime: Current operational regime
            anomaly_score: Anomaly score
            dynamic_features: Dynamic features
            historical_patterns: Historical patterns from memory
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Regime-specific recommendations
        if regime == "STARTUP":
            recommendations.append("Monitorear ramp-up de temperatura")
            recommendations.append("Verificar estabilidad de presión")
        elif regime == "SHUTDOWN":
            recommendations.append("Monitorear cooldown de sistema")
            recommendations.append("Verificar secuencia de apagado")
        elif regime == "VOLATILE_PEAK":
            recommendations.append("Verificar carga operacional")
            recommendations.append("Revisar estabilidad térmica")
        elif regime == "STABLE_NORMAL":
            recommendations.append("Continuar monitoreo normal")
        
        # Anomaly-specific recommendations
        if anomaly_score > 0.8:
            recommendations.append("Investigar anomalía de alta prioridad")
            recommendations.append("Verificar correlación con otros sensores")
        elif anomaly_score > 0.6:
            recommendations.append("Monitorear anomalía de prioridad media")
        
        # Dynamic feature-specific recommendations
        derivative = dynamic_features.get("derivative", 0.0)
        if abs(derivative) > 2.0:
            recommendations.append("Verificar tasa de cambio acelerada")
        
        rolling_std = dynamic_features.get("rolling_std_1h", 0.0)
        if rolling_std > 5.0:
            recommendations.append("Revisar volatilidad operacional")
        
        # Historical pattern-based recommendations
        if "STARTUP" in " ".join(historical_patterns):
            recommendations.append("Validar comportamiento durante startup")
        
        if "VOLATILE_PEAK" in " ".join(historical_patterns):
            recommendations.append("Preparar para picos de carga")
        
        return recommendations
