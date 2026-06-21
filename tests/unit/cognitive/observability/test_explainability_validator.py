"""
Unit tests for ExplainabilityValidator.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.observability.explainability_validator import ExplainabilityValidator
from domain.entities.explainability import ContextualExplanation


class TestExplainabilityValidator(unittest.TestCase):
    """Test cases for ExplainabilityValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ExplainabilityValidator()
    
    def test_validate_explanation_basic(self):
        """Test basic explanation validation."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2},
            similar_event_count=3,
            historical_context="3 eventos similares",
            historical_patterns=["STARTUP"],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up"],
        )
        
        result = self.validator.validate_explanation(explanation, retrieval_relevance=0.8)
        
        self.assertIn("temporal_consistency", result)
        self.assertIn("confidence_coherence", result)
        self.assertIn("retrieval_relevance", result)
        self.assertIn("stability_score", result)
        self.assertIn("explainability_quality_score", result)
    
    def test_validate_explanation_no_history(self):
        """Test explanation validation with no history."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2},
            similar_event_count=0,
            historical_context="No historical similar events",
            historical_patterns=[],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up"],
        )
        
        result = self.validator.validate_explanation(explanation, retrieval_relevance=0.8)
        
        # With no history, temporal consistency should be 1.0
        self.assertEqual(result["temporal_consistency"], 1.0)
    
    def test_validate_confidence_coherence(self):
        """Test confidence coherence validation."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2},
            similar_event_count=0,
            historical_context="No historical similar events",
            historical_patterns=[],
            operational_confidence=0.88,  # Close to expected
            suggested_actions=["Monitorear ramp-up"],
        )
        
        result = self.validator.validate_explanation(explanation, retrieval_relevance=0.8)
        
        self.assertGreater(result["confidence_coherence"], 0.8)
    
    def test_validate_stability(self):
        """Test stability validation."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2},
            similar_event_count=0,
            historical_context="No historical similar events",
            historical_patterns=[],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up"],
        )
        
        result = self.validator.validate_explanation(explanation, retrieval_relevance=0.8)
        
        self.assertEqual(result["stability_score"], 1.0)
    
    def test_detect_contradictions(self):
        """Test contradiction detection."""
        exp1 = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2},
            similar_event_count=0,
            historical_context="No historical similar events",
            historical_patterns=[],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up"],
        )
        
        exp2 = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567891.0,
            current_regime="STABLE_NORMAL",  # Different regime
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2},
            similar_event_count=0,
            historical_context="No historical similar events",
            historical_patterns=[],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up"],
        )
        
        contradictions = self.validator.detect_contradictions([exp1, exp2])
        
        self.assertGreater(len(contradictions), 0)
        self.assertEqual(contradictions[0]["type"], "regime_contradiction")
    
    def test_reset_history(self):
        """Test resetting explanation history."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2},
            similar_event_count=0,
            historical_context="No historical similar events",
            historical_patterns=[],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up"],
        )
        
        self.validator.validate_explanation(explanation, retrieval_relevance=0.8)
        self.validator.reset_history()
        
        # After reset, temporal consistency should be 1.0 again
        result = self.validator.validate_explanation(explanation, retrieval_relevance=0.8)
        self.assertEqual(result["temporal_consistency"], 1.0)


if __name__ == "__main__":
    unittest.main()
