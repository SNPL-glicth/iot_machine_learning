"""Tests for PatternInterpreter."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.pattern_interpreter import PatternInterpreter
from iot_machine_learning.infrastructure.ml.cognitive.pattern_interpreter.types import InterpretedPattern


class TestPatternInterpreter:
    """Test suite for PatternInterpreter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.interpreter = PatternInterpreter()
    
    def test_interpret_text_patterns_with_urgency(self):
        """Test text pattern interpretation with high urgency."""
        raw_patterns = {
            "pattern_summary": {
                "has_escalation": True,
                "urgency_trend": "increasing",
                "n_critical_spikes": 2,
                "urgency_regime": "high",
                "improvement_points": 0,
            },
            "change_points": [{"index": 10, "magnitude": 1.5}],
            "spikes": [{"magnitude": 3.0, "position": 5}],
        }
        
        result = self.interpreter.interpret(
            raw_patterns=raw_patterns,
            input_type="text",
            domain="infrastructure",
            urgency_score=0.8,
            sentiment_label="negative"
        )
        
        assert len(result) > 0
        assert all(isinstance(p, InterpretedPattern) for p in result)
        assert any(p.severity_hint == "critical" for p in result)
        assert all(p.data_type == "text" for p in result)
    
    def test_interpret_text_patterns_stable(self):
        """Test text pattern interpretation with stable operations."""
        raw_patterns = {
            "pattern_summary": {
                "n_change_points": 0,
                "n_spikes": 0,
            },
            "change_points": [],
            "spikes": [],
        }
        
        result = self.interpreter.interpret(
            raw_patterns=raw_patterns,
            input_type="text",
            domain="operations",
            urgency_score=0.1,
            sentiment_label="neutral"
        )
        
        assert len(result) > 0
        assert any(p.pattern_type == "stable_operations" for p in result)
        assert all(p.severity_hint == "info" for p in result)
    
    def test_interpret_numeric_patterns_with_spikes(self):
        """Test numeric pattern interpretation with anomalous spikes."""
        raw_patterns = {
            "delta_spikes": [
                {"is_delta_spike": True, "delta_magnitude": 3.5, "confidence": 0.9}
            ],
            "change_points": [
                {"change_type": "level_shift", "magnitude": 4.0, "confidence": 0.8}
            ],
            "pattern_result": type('PatternResult', (), {
                'pattern_type': type('PatternType', (), {'value': 'spike'})(),
                'confidence': 0.85
            })()
        }
        
        result = self.interpreter.interpret(
            raw_patterns=raw_patterns,
            input_type="numeric",
            domain="security",
            urgency_score=0.6,
        )
        
        assert len(result) > 0
        assert any(p.severity_hint == "critical" for p in result)
        assert all(p.data_type == "numeric" for p in result)
    
    def test_interpret_numeric_patterns_stable(self):
        """Test numeric pattern interpretation with stable operations."""
        raw_patterns = {
            "delta_spikes": [],
            "change_points": [],
            "pattern_result": type('PatternResult', (), {
                'pattern_type': type('PatternType', (), {'value': 'stable'})(),
                'confidence': 0.8
            })()
        }
        
        result = self.interpreter.interpret(
            raw_patterns=raw_patterns,
            input_type="numeric",
            domain="infrastructure",
            urgency_score=0.1,
        )
        
        assert len(result) > 0
        assert any(p.pattern_type == "stable" for p in result)
        assert all(p.severity_hint == "info" for p in result)
    
    def test_interpret_universal_patterns(self):
        """Test universal pattern interpretation (merges text + numeric)."""
        raw_patterns = {
            "pattern_summary": {
                "has_escalation": True,
                "urgency_regime": "high",
            },
            "delta_spikes": [
                {"is_delta_spike": True, "delta_magnitude": 2.5}
            ],
            "change_points": [{"magnitude": 1.5}],
            "spikes": [{"magnitude": 2.0}],
        }
        
        result = self.interpreter.interpret(
            raw_patterns=raw_patterns,
            input_type="universal",
            domain="infrastructure",
            urgency_score=0.7,
        )
        
        assert len(result) > 0
        # Should have patterns from both text and numeric interpretation
        data_types = {p.data_type for p in result}
        assert "text" in data_types or "numeric" in data_types
    
    def test_get_primary_pattern(self):
        """Test getting the most severe pattern."""
        patterns = [
            InterpretedPattern(
                pattern_type="stable",
                short_name="Estable",
                description="Operación estable",
                severity_hint="info",
                domain_context="Normal",
                confidence=0.8,
                data_type="numeric"
            ),
            InterpretedPattern(
                pattern_type="anomalous_spike",
                short_name="Spike anómalo",
                description="Valor atípico",
                severity_hint="critical",
                domain_context="Requiere atención",
                confidence=0.9,
                data_type="numeric"
            ),
            InterpretedPattern(
                pattern_type="drift_detected",
                short_name="Deriva",
                description="Desviación gradual",
                severity_hint="warning",
                domain_context="Monitorear",
                confidence=0.7,
                data_type="numeric"
            ),
        ]
        
        primary = self.interpreter.get_primary_pattern(patterns)
        
        assert primary is not None
        assert primary.severity_hint == "critical"
        assert primary.pattern_type == "anomalous_spike"
    
    def test_get_primary_pattern_empty(self):
        """Test getting primary pattern from empty list."""
        primary = self.interpreter.get_primary_pattern([])
        assert primary is None
    
    def test_format_for_conclusion(self):
        """Test formatting patterns for conclusion."""
        patterns = [
            InterpretedPattern(
                pattern_type="anomalous_spike",
                short_name="Spike anómalo",
                description="Valor atípico detectado",
                severity_hint="critical",
                domain_context="Requiere verificación inmediata",
                confidence=0.9,
                data_type="numeric"
            ),
            InterpretedPattern(
                pattern_type="drift_detected",
                short_name="Deriva",
                description="Desviación gradual",
                severity_hint="critical",
                domain_context="Monitorear tendencia",
                confidence=0.7,
                data_type="numeric"
            ),
        ]
        
        conclusion = self.interpreter.format_for_conclusion(patterns, "infrastructure")
        
        assert "Spike anómalo" in conclusion
        assert "Contexto:" in conclusion
        assert "Confianza: 90%" in conclusion
        assert "Otros patrones críticos" in conclusion
    
    def test_format_for_conclusion_empty(self):
        """Test formatting conclusion with no patterns."""
        conclusion = self.interpreter.format_for_conclusion([], "infrastructure")
        assert "No se detectaron patrones" in conclusion
    
    def test_get_pattern_summary(self):
        """Test getting pattern summary statistics."""
        patterns = [
            InterpretedPattern("stable", "Estable", "Normal", "info", "OK", 0.8, "text"),
            InterpretedPattern("anomalous_spike", "Spike", "Anómalo", "critical", "Check", 0.9, "numeric"),
            InterpretedPattern("drift_detected", "Deriva", "Gradual", "warning", "Watch", 0.7, "numeric"),
        ]
        
        summary = self.interpreter.get_pattern_summary(patterns)
        
        assert summary["total_patterns"] == 3
        assert summary["severity_breakdown"]["critical"] == 1
        assert summary["severity_breakdown"]["warning"] == 1
        assert summary["severity_breakdown"]["info"] == 1
        assert set(summary["data_types"]) == {"text", "numeric"}
        assert summary["primary_pattern"] is not None
        assert summary["primary_pattern"].severity_hint == "critical"
    
    def test_get_pattern_summary_empty(self):
        """Test getting summary from empty pattern list."""
        summary = self.interpreter.get_pattern_summary([])
        
        assert summary["total_patterns"] == 0
        assert summary["severity_breakdown"]["critical"] == 0
        assert summary["severity_breakdown"]["warning"] == 0
        assert summary["severity_breakdown"]["info"] == 0
        assert summary["data_types"] == []
        assert summary["primary_pattern"] is None
    
    def test_domain_context_enrichment(self):
        """Test that domain context is properly enriched."""
        raw_patterns = {
            "pattern_summary": {"has_escalation": True},
            "delta_spikes": [{"is_delta_spike": True, "delta_magnitude": 3.0}],
        }
        
        # Test infrastructure domain
        result_infra = self.interpreter.interpret(
            raw_patterns, "text", "infrastructure", urgency_score=0.8
        )
        
        # Test security domain
        result_security = self.interpreter.interpret(
            raw_patterns, "text", "security", urgency_score=0.8
        )
        
        # Both should have patterns but with different domain contexts
        assert len(result_infra) > 0
        assert len(result_security) > 0
        
        # Check that domain contexts are different
        infra_contexts = [p.domain_context for p in result_infra]
        security_contexts = [p.domain_context for p in result_security]
        
        # Should contain domain-specific information
        assert any("infrastructure" in ctx.lower() or "revisa" in ctx.lower() for ctx in infra_contexts)
        assert any("security" in ctx.lower() or "ataque" in ctx.lower() for ctx in security_contexts)
    
    def test_graceful_failure(self):
        """Test graceful failure when pattern interpretation fails."""
        # Invalid input type
        result = self.interpreter.interpret(
            raw_patterns={},
            input_type="invalid_type",
            domain="infrastructure"
        )
        
        assert result == []
    
    def test_confidence_calculation(self):
        """Test confidence values are properly calculated."""
        raw_patterns = {
            "pattern_summary": {"has_escalation": True},
            "spikes": [{"magnitude": 5.0}],
        }
        
        result = self.interpreter.interpret(
            raw_patterns, "text", "infrastructure", urgency_score=0.9
        )
        
        for pattern in result:
            assert 0.0 <= pattern.confidence <= 1.0
            # Higher urgency should result in higher confidence for critical patterns
            if pattern.severity_hint == "critical":
                assert pattern.confidence > 0.5
