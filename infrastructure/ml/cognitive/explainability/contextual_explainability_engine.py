"""
ContextualExplainabilityEngine for generating contextual explanations.

Generates structured contextual explanations using operational memory.
"""

from typing import List, Dict, Any, Optional

from domain.entities.explainability import ContextualExplanation
from infrastructure.ml.cognitive.memory.historical_similarity_retriever import HistoricalSimilarityRetriever
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry
from .historical_context_aggregator import HistoricalContextAggregator
from .recommendation_generator import RecommendationGenerator
from .contextual_confidence_calculator import ContextualConfidenceCalculator


class ContextualExplainabilityEngine:
    """Engine for generating contextual explanations with memory."""
    
    def __init__(
        self,
        similarity_retriever: HistoricalSimilarityRetriever,
        registry: CognitiveMemoryRegistry,
    ):
        """
        Initialize explainability engine.
        
        Args:
            similarity_retriever: Historical similarity retriever
            registry: Cognitive memory registry
        """
        self._similarity_retriever = similarity_retriever
        self._registry = registry
        self._context_aggregator = HistoricalContextAggregator()
        self._recommendation_generator = RecommendationGenerator()
        self._confidence_calculator = ContextualConfidenceCalculator()
    
    def generate_explanation(
        self,
        sensor_id: int,
        sensor_type: str,
        ml_features: Dict[str, Any],
        regime: str,
        anomaly_score: float,
        regime_confidence: float = 0.8,
    ) -> ContextualExplanation:
        """
        Generate contextual explanation.
        
        Args:
            sensor_id: Sensor identifier
            sensor_type: Sensor type
            ml_features: ML features dictionary
            regime: Current operational regime
            anomaly_score: Anomaly score
            regime_confidence: Regime classification confidence
        
        Returns:
            ContextualExplanation with structured explanation
        """
        # 1. Identify primary drivers
        primary_drivers = self._identify_primary_drivers(ml_features, anomaly_score)
        
        # 2. Extract dynamic context
        dynamic_context = self._extract_dynamic_context(ml_features)
        
        # 3. Retrieve similar historical events
        similar_events = []
        if self._registry.enable_retrieval:
            similar_events = self._similarity_retriever.retrieve(
                sensor_id=sensor_id,
                ml_features=ml_features,
                regime=regime,
                top_k=5,
            )
        
        # 4. Aggregate historical context
        historical_aggregation = self._context_aggregator.aggregate(similar_events)
        
        # 5. Generate recommendations
        recommendations = self._recommendation_generator.generate(
            regime=regime,
            anomaly_score=anomaly_score,
            dynamic_features=ml_features.get("dynamic_features", {}),
            historical_patterns=historical_aggregation["historical_patterns"],
        )
        
        # 6. Calculate contextual confidence
        retrieval_similarity = 0.0 if not similar_events else 0.8  # Simplified
        feature_stability = ml_features.get("stability", 0.5)
        operational_confidence = self._confidence_calculator.calculate(
            anomaly_score=anomaly_score,
            retrieval_similarity=retrieval_similarity,
            regime_confidence=regime_confidence,
            feature_stability=feature_stability,
        )
        
        return ContextualExplanation(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            timestamp=ml_features.get("timestamp", 0.0),
            current_regime=regime,
            anomaly_score=anomaly_score,
            primary_drivers=primary_drivers,
            dynamic_context=dynamic_context,
            similar_event_count=historical_aggregation["similar_event_count"],
            historical_context=historical_aggregation["historical_context"],
            historical_patterns=historical_aggregation["historical_patterns"],
            operational_confidence=operational_confidence,
            suggested_actions=recommendations,
        )
    
    def _identify_primary_drivers(
        self,
        ml_features: Dict[str, Any],
        anomaly_score: float,
    ) -> List[str]:
        """Identify primary drivers of anomaly."""
        drivers = []
        
        z_score = ml_features.get("z_score", 0.0)
        if abs(z_score) > 2.0:
            drivers.append(f"Desviación Z-score ({z_score:.2f}σ)")
        
        dynamic_features = ml_features.get("dynamic_features", {})
        derivative = dynamic_features.get("derivative", 0.0)
        if abs(derivative) > 1.0:
            drivers.append(f"Tasa de cambio ({derivative:.2f})")
        
        rolling_std = dynamic_features.get("rolling_std_1h", 0.0)
        if rolling_std > 3.0:
            drivers.append(f"Volatilidad ({rolling_std:.2f})")
        
        if not drivers:
            drivers.append("Comportamiento atípico general")
        
        return drivers
    
    def _extract_dynamic_context(self, ml_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dynamic context from ML features."""
        dynamic_features = ml_features.get("dynamic_features", {})
        
        return {
            "current_value": ml_features.get("current_value", 0.0),
            "baseline": ml_features.get("baseline", 0.0),
            "z_score": ml_features.get("z_score", 0.0),
            "trend": ml_features.get("trend", "unknown"),
            "stability": ml_features.get("stability", 0.0),
            "derivative": dynamic_features.get("derivative"),
            "rolling_std_1h": dynamic_features.get("rolling_std_1h"),
        }
