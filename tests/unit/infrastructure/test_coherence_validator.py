"""Tests for CoherenceValidator."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.universal.validation.coherence_validator import (
    CoherenceValidator,
    CoherenceReport,
)
from iot_machine_learning.domain.services.severity_rules import SeverityResult


@dataclass
class MockSeverity:
    """Mock severity object."""
    severity: str
    risk_level: str = "MEDIUM"


@dataclass
class MockResult:
    """Mock analysis result for testing."""
    severity: Any
    confidence: float
    analysis: Dict[str, Any]
    patterns: List[Any] = None
    explanation: Any = None
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = []


class TestCoherenceValidator:
    """Test suite for CoherenceValidator."""
    
    def test_coherent_result_passes(self):
        """Test that coherent result passes validation."""
        validator = CoherenceValidator()
        
        result = MockResult(
            severity=MockSeverity("warning"),
            confidence=0.75,
            analysis={
                "urgency_score": 0.6,
                "sentiment_label": "neutral",
            }
        )
        
        report = validator.validate(result)
        
        assert report.is_coherent is True
        assert len(report.warnings) == 0
        assert len(report.adjustments) == 0
    
    def test_rule1_high_urgency_positive_sentiment(self):
        """Test Rule 1: High urgency + positive sentiment."""
        validator = CoherenceValidator()
        
        result = MockResult(
            severity=MockSeverity("warning"),
            confidence=0.75,
            analysis={
                "urgency_score": 0.85,
                "sentiment_label": "positive",
            }
        )
        
        report = validator.validate(result)
        
        assert report.is_coherent is False
        assert len(report.warnings) == 1
        assert "urgency" in report.warnings[0].lower()
        assert "positive sentiment" in report.warnings[0].lower()
        assert len(report.adjustments) == 1
        assert result.analysis["urgency_score"] == 0.6  # Lowered
    
    def test_rule2_critical_severity_low_confidence(self):
        """Test Rule 2: Critical severity + low confidence."""
        validator = CoherenceValidator()
        
        result = MockResult(
            severity=MockSeverity("critical"),
            confidence=0.35,
            analysis={}
        )
        
        report = validator.validate(result)
        
        assert report.is_coherent is False
        assert len(report.warnings) == 1
        assert "critical" in report.warnings[0].lower() or "high" in report.warnings[0].lower()
        assert "confidence" in report.warnings[0].lower()
        assert len(report.adjustments) == 1
        assert result.severity.severity == "warning"  # Downgraded
    
    def test_rule3_critical_actions_stable_pattern(self):
        """Test Rule 3: Critical actions + stable pattern."""
        validator = CoherenceValidator()
        
        @dataclass
        class MockPattern:
            pattern_type: str
        
        result = MockResult(
            severity=MockSeverity("critical"),
            confidence=0.75,
            analysis={
                "conclusion": "Stop production immediately",
            },
            patterns=[MockPattern("stable_operation")]
        )
        
        report = validator.validate(result)
        
        assert report.is_coherent is False
        assert len(report.warnings) == 1
        assert "critical actions" in report.warnings[0].lower()
        assert "stable" in report.warnings[0].lower()
        assert len(report.adjustments) == 1
        assert result.severity.severity == "warning"  # Downgraded
    
    def test_rule4_multiple_confidence_values(self):
        """Test Rule 4: Multiple confidence values."""
        validator = CoherenceValidator()
        
        @dataclass
        class MockOutcome:
            confidence: float
        
        @dataclass
        class MockExplanation:
            outcome: MockOutcome
        
        result = MockResult(
            severity=MockSeverity("warning"),
            confidence=0.60,
            analysis={
                "confidence": 0.80,
            },
            explanation=MockExplanation(MockOutcome(0.75))
        )
        
        report = validator.validate(result)
        
        assert report.is_coherent is False
        assert len(report.warnings) == 1
        assert "multiple confidence" in report.warnings[0].lower()
        assert len(report.adjustments) == 1
        assert result.confidence == 0.80  # Unified to highest
    
    def test_rule5_critical_severity_low_urgency(self):
        """Test Rule 5: Critical severity + low urgency."""
        validator = CoherenceValidator()
        
        result = MockResult(
            severity=MockSeverity("critical"),
            confidence=0.75,
            analysis={
                "urgency_score": 0.15,
            }
        )
        
        report = validator.validate(result)
        
        assert report.is_coherent is False
        assert len(report.warnings) == 1
        assert "critical" in report.warnings[0].lower() or "high" in report.warnings[0].lower()
        assert "urgency" in report.warnings[0].lower()
        assert len(report.adjustments) == 1
        assert result.severity.severity == "warning"  # Downgraded
    
    def test_multiple_violations(self):
        """Test multiple coherence violations."""
        validator = CoherenceValidator()
        
        result = MockResult(
            severity=MockSeverity("critical"),
            confidence=0.35,
            analysis={
                "urgency_score": 0.85,
                "sentiment_label": "positive",
            }
        )
        
        report = validator.validate(result)
        
        assert report.is_coherent is False
        assert len(report.warnings) >= 2  # At least 2 violations
        assert len(report.adjustments) >= 2
    
    def test_coherence_warnings_added_to_result(self):
        """Test that coherence warnings are added to result."""
        validator = CoherenceValidator()
        
        result = MockResult(
            severity=MockSeverity("critical"),
            confidence=0.35,
            analysis={}
        )
        
        report = validator.validate(result)
        
        assert "coherence_warnings" in result.analysis
        assert "coherence_adjustments" in result.analysis
        assert len(result.analysis["coherence_warnings"]) > 0
        assert len(result.analysis["coherence_adjustments"]) > 0
    
    def test_conservative_severity_downgrade(self):
        """Test that severity is always downgraded conservatively."""
        validator = CoherenceValidator()
        
        # Test high severity with low confidence
        result = MockResult(
            severity=MockSeverity("high"),
            confidence=0.40,
            analysis={}
        )
        
        report = validator.validate(result)
        
        # Should downgrade to warning (more conservative)
        assert result.severity.severity == "warning"
    
    def test_no_severity_escalation(self):
        """Test that severity is never escalated."""
        validator = CoherenceValidator()
        
        # Start with low severity
        result = MockResult(
            severity=MockSeverity("low"),
            confidence=0.95,
            analysis={
                "urgency_score": 0.95,
                "sentiment_label": "negative",
            }
        )
        
        report = validator.validate(result)
        
        # Severity should remain low or be downgraded, never escalated
        assert result.severity.severity in ["info", "low"]
    
    def test_report_to_dict(self):
        """Test CoherenceReport.to_dict()."""
        report = CoherenceReport(
            is_coherent=False,
            warnings=["Warning 1", "Warning 2"],
            adjustments=["Adjustment 1"]
        )
        
        d = report.to_dict()
        
        assert d["is_coherent"] is False
        assert len(d["warnings"]) == 2
        assert len(d["adjustments"]) == 1
