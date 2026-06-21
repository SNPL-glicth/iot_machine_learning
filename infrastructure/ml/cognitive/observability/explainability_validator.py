"""
ExplainabilityValidator for validating explainability quality.

Validates temporal consistency, contradiction detection, confidence coherence, retrieval relevance, and explainability stability.
"""

from typing import List, Dict, Any, Tuple
import time
import statistics

from domain.entities.explainability import ContextualExplanation


class ExplainabilityValidator:
    """Validator for explainability quality."""
    
    def __init__(
        self,
        consistency_threshold: float = 0.8,
        confidence_coherence_threshold: float = 0.7,
        retrieval_relevance_threshold: float = 0.6,
    ):
        """
        Initialize explainability validator.
        
        Args:
            consistency_threshold: Threshold for temporal consistency
            confidence_coherence_threshold: Threshold for confidence coherence
            retrieval_relevance_threshold: Threshold for retrieval relevance
        """
        self._consistency_threshold = consistency_threshold
        self._confidence_coherence_threshold = confidence_coherence_threshold
        self._retrieval_relevance_threshold = retrieval_relevance_threshold
        
        # Historical explanations for consistency checking
        self._explanation_history: List[Tuple[float, ContextualExplanation]] = []
    
    def validate_explanation(
        self,
        explanation: ContextualExplanation,
        retrieval_relevance: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Validate explanation quality.
        
        Args:
            explanation: ContextualExplanation to validate
            retrieval_relevance: Retrieval relevance score [0, 1]
        
        Returns:
            Dictionary with validation results
        """
        # Validate temporal consistency
        temporal_consistency = self._validate_temporal_consistency(explanation)
        
        # Validate confidence coherence
        confidence_coherence = self._validate_confidence_coherence(explanation)
        
        # Validate retrieval relevance
        retrieval_relevance_valid = retrieval_relevance >= self._retrieval_relevance_threshold
        
        # Validate explainability stability
        stability_score = self._validate_stability(explanation)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            temporal_consistency,
            confidence_coherence,
            retrieval_relevance,
            stability_score,
        )
        
        # Store explanation in history
        self._explanation_history.append((time.time(), explanation))
        
        # Keep history limited to last 100 explanations
        if len(self._explanation_history) > 100:
            self._explanation_history = self._explanation_history[-100:]
        
        return {
            "temporal_consistency": temporal_consistency,
            "confidence_coherence": confidence_coherence,
            "retrieval_relevance": retrieval_relevance,
            "retrieval_relevance_valid": retrieval_relevance_valid,
            "stability_score": stability_score,
            "explainability_quality_score": quality_score,
            "validation_timestamp": time.time(),
        }
    
    def _validate_temporal_consistency(self, explanation: ContextualExplanation) -> float:
        """Validate temporal consistency with historical explanations."""
        if not self._explanation_history:
            return 1.0  # No history, assume consistent
        
        # Find similar explanations from history
        similar_explanations = [
            exp for _, exp in self._explanation_history
            if exp.sensor_id == explanation.sensor_id
            and exp.sensor_type == explanation.sensor_type
        ]
        
        if not similar_explanations:
            return 1.0  # No similar explanations, assume consistent
        
        # Compare primary drivers
        driver_consistency_scores = []
        for hist_exp in similar_explanations[-10:]:  # Last 10 similar explanations
            driver_overlap = len(set(explanation.primary_drivers) & set(hist_exp.primary_drivers))
            driver_consistency = driver_overlap / max(len(explanation.primary_drivers), 1)
            driver_consistency_scores.append(driver_consistency)
        
        return statistics.mean(driver_consistency_scores) if driver_consistency_scores else 1.0
    
    def _validate_confidence_coherence(self, explanation: ContextualExplanation) -> float:
        """Validate confidence coherence with anomaly score."""
        # Confidence should correlate with anomaly score
        # Higher anomaly score should generally have higher confidence
        expected_confidence = explanation.anomaly_score * 0.8 + 0.2  # Baseline
        confidence_diff = abs(explanation.operational_confidence - expected_confidence)
        coherence = max(0.0, 1.0 - confidence_diff)
        
        return coherence
    
    def _validate_stability(self, explanation: ContextualExplanation) -> float:
        """Validate explainability stability."""
        # Check if explanation has all required fields
        required_fields = [
            explanation.sensor_id,
            explanation.sensor_type,
            explanation.current_regime,
            explanation.anomaly_score,
            explanation.primary_drivers,
            explanation.suggested_actions,
        ]
        
        stability = 1.0 if all(field is not None and field != [] for field in required_fields) else 0.0
        
        return stability
    
    def _calculate_quality_score(
        self,
        temporal_consistency: float,
        confidence_coherence: float,
        retrieval_relevance: float,
        stability_score: float,
    ) -> float:
        """Calculate overall explainability quality score."""
        weights = {
            "temporal_consistency": 0.3,
            "confidence_coherence": 0.2,
            "retrieval_relevance": 0.3,
            "stability": 0.2,
        }
        
        quality_score = (
            weights["temporal_consistency"] * temporal_consistency +
            weights["confidence_coherence"] * confidence_coherence +
            weights["retrieval_relevance"] * retrieval_relevance +
            weights["stability"] * stability_score
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def detect_contradictions(
        self,
        explanations: List[ContextualExplanation],
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions between explanations.
        
        Args:
            explanations: List of explanations to check
        
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        for i, exp1 in enumerate(explanations):
            for exp2 in explanations[i+1:]:
                if exp1.sensor_id == exp2.sensor_id:
                    # Check for contradictory regimes
                    if exp1.current_regime != exp2.current_regime:
                        contradictions.append({
                            "type": "regime_contradiction",
                            "sensor_id": exp1.sensor_id,
                            "regime1": exp1.current_regime,
                            "regime2": exp2.current_regime,
                            "timestamp1": exp1.timestamp,
                            "timestamp2": exp2.timestamp,
                        })
                    
                    # Check for contradictory anomaly scores
                    score_diff = abs(exp1.anomaly_score - exp2.anomaly_score)
                    if score_diff > 0.5:
                        contradictions.append({
                            "type": "anomaly_score_contradiction",
                            "sensor_id": exp1.sensor_id,
                            "score1": exp1.anomaly_score,
                            "score2": exp2.anomaly_score,
                            "timestamp1": exp1.timestamp,
                            "timestamp2": exp2.timestamp,
                        })
        
        return contradictions
    
    def reset_history(self) -> None:
        """Reset explanation history."""
        self._explanation_history = []
