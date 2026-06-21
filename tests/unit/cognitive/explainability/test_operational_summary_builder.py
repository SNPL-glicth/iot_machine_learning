"""
Unit tests for OperationalSummaryBuilder.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import unittest

from infrastructure.ml.cognitive.explainability.operational_summary_builder import OperationalSummaryBuilder
from domain.entities.explainability import ContextualExplanation


class TestOperationalSummaryBuilder(unittest.TestCase):
    """Test cases for OperationalSummaryBuilder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = OperationalSummaryBuilder()
    
    def test_build_summary_basic(self):
        """Test building basic summary."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)", "Tasa de cambio (2.50)"],
            dynamic_context={"current_value": 85.2, "baseline": 45.0},
            similar_event_count=3,
            historical_context="3 eventos similares encontrados durante STARTUP.",
            historical_patterns=["Patrones recurrentes en régimen STARTUP"],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up de temperatura", "Verificar estabilidad de presión"],
        )
        
        summary = self.builder.build(explanation)
        
        self.assertIn("sensor_id", summary)
        self.assertIn("sensor_type", summary)
        self.assertIn("current_regime", summary)
        self.assertIn("anomaly_score", summary)
        self.assertIn("operational_confidence", summary)
        self.assertIn("primary_drivers", summary)
        self.assertIn("similar_event_count", summary)
        self.assertIn("suggested_actions", summary)
        self.assertIn("summary_text", summary)
    
    def test_build_summary_text_anomaly(self):
        """Test building summary text for anomaly."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2, "baseline": 45.0},
            similar_event_count=3,
            historical_context="3 eventos similares encontrados.",
            historical_patterns=[],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up"],
        )
        
        summary = self.builder.build(explanation)
        
        self.assertIn("Anomalía detectada", summary["summary_text"])
        self.assertIn("0.85", summary["summary_text"])
        self.assertIn("STARTUP", summary["summary_text"])
    
    def test_build_summary_text_normal(self):
        """Test building summary text for normal operation."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STABLE_NORMAL",
            anomaly_score=0.3,
            primary_drivers=["Comportamiento atípico general"],
            dynamic_context={"current_value": 45.0, "baseline": 44.0},
            similar_event_count=0,
            historical_context="No historical similar events found.",
            historical_patterns=[],
            operational_confidence=0.5,
            suggested_actions=["Continuar monitoreo normal"],
        )
        
        summary = self.builder.build(explanation)
        
        self.assertIn("Operación normal", summary["summary_text"])
        self.assertIn("0.30", summary["summary_text"])
    
    def test_build_summary_text_with_historical_context(self):
        """Test building summary text with historical context."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2, "baseline": 45.0},
            similar_event_count=5,
            historical_context="5 eventos similares encontrados. La mayoría ocurrieron durante STARTUP.",
            historical_patterns=["Patrones recurrentes en régimen STARTUP"],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up"],
        )
        
        summary = self.builder.build(explanation)
        
        self.assertIn("5 eventos similares", summary["summary_text"])
    
    def test_build_summary_text_with_recommendations(self):
        """Test building summary text with recommendations."""
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)"],
            dynamic_context={"current_value": 85.2, "baseline": 45.0},
            similar_event_count=0,
            historical_context="No historical similar events found.",
            historical_patterns=[],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up de temperatura", "Verificar estabilidad de presión"],
        )
        
        summary = self.builder.build(explanation)
        
        self.assertIn("Acciones sugeridas", summary["summary_text"])
        self.assertIn("Monitorear ramp-up", summary["summary_text"])


if __name__ == "__main__":
    unittest.main()
